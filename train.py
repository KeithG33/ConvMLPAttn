
import os
import time
import json
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, random_split, default_collate
import torchvision.transforms.v2 as transforms

from tqdm import tqdm
import wandb

from accelerate import Accelerator
from datasets import load_dataset

from in_dataset import inDataset
from model import MLPAttn2DNet


torch.backends.cudnn.benchmark = True


class EarlyStopping:
    def __init__(self, patience=7, min_delta=0):
        self.patience = patience
        self.min_delta = min_delta
        self.counter = 0
        self.best_loss = None
        self.early_stop = False

    def __call__(self, val_loss):
        if self.best_loss is None:
            self.best_loss = val_loss

        elif val_loss > self.best_loss - self.min_delta:
            self.counter += 1
            if self.counter >= self.patience:
                print(f"Early stopping after {self.patience} epochs.")
                self.early_stop = True
        else:
            self.best_loss = val_loss
            self.counter = 0
 
        return self.early_stop
    

class Trainer:
    def __init__(
        self,
        epochs=50,
        batch_size=128,
        lr=1e-4,
        save_dir='./checkpoints',
        wandb_logging=False,
        log_every=50,
        wandb_name='IN1k-Training', 
        wandb_id=None,
        checkpoint_dir=None,
    ):
        self.epochs = epochs
        self.batch_size = batch_size
        self.lr = lr
        self.save_dir = save_dir
        self.log_every = log_every

        # CIFAR100 settings
        n_channels = 3
        n_classes = 1000
        
        self.accelerator = Accelerator(
            mixed_precision='bf16',
            dynamo_backend='INDUCTOR',
        )

        self.early_stopper = EarlyStopping(patience=100, min_delta=0.0)

        # Optionally initialize wandb logging
        self.wandb_logging = wandb_logging
        if wandb_logging:
            resume = 'must' if wandb_id else None
            wandb.init(project="IN1k-training", name=wandb_name, id=wandb_id, resume=resume)

        # Define transforms for training
        self.train_transform = transforms.Compose([
            transforms.RandomResizedCrop(224, scale=(0.5, 1.0)),
            transforms.TrivialAugmentWide(),
            transforms.RandomHorizontalFlip(),
            transforms.RandomErasing(p=0.1, value='random', scale=(0.02, 0.1)),
            transforms.RandomApply([
                transforms.GaussianBlur(kernel_size=3),
            ], p=0.1),      
            transforms.ToImage(),
            transforms.ToDtype(torch.float32, scale=True),
            transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                 std=[0.229, 0.224, 0.225])
        ])

        # Define transforms for validation and testing (no augmentation)
        self.test_transform = transforms.Compose([
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToImage(),
            transforms.ToDtype(torch.float32, scale=True),
            transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                 std=[0.229, 0.224, 0.225])
        ])
        
        cutmix = transforms.CutMix(num_classes=n_classes)
        # mixup = transforms.MixUp(num_classes=n_classes)
        cutmix_or_mixup = transforms.RandomChoice([
            transforms.RandomApply([cutmix], p=0.1), 
            # transforms.RandomApply([mixup], p=0.2)
        ])
        def collate_fn(batch):
            return cutmix_or_mixup(*default_collate(batch))

        self.train_dataset = inDataset(
            dataset='train',
            train_transforms=self.train_transform,
        )
        self.val_dataset = inDataset(
            dataset='validation',
            val_transforms=self.test_transform,
        )

        # Create DataLoader for training and validation
        self.train_loader = DataLoader(
            self.train_dataset, 
            batch_size=self.batch_size, 
            shuffle=True, 
            num_workers=8,
            collate_fn=collate_fn,
            pin_memory=True,

        )
        self.val_loader = DataLoader(
            self.val_dataset, 
            batch_size=self.batch_size, 
            shuffle=False, 
            num_workers=8,
            pin_memory=True,
        )
        
        self.model = MLPAttn2DNet(
            depths=(2, 2, 6, 3),
            dims=(48, 96, 128, 224),
            heads=(2, 2, 4, 8),
            mlp_expand=2,
        )
        print(f"Model parameters: {sum(p.numel() for p in self.model.parameters())}")

        # Set device and move model to device
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model.to(self.device)

        # Use CrossEntropyLoss for single-label classification with label smoothing
        self.criterion = nn.CrossEntropyLoss(label_smoothing=0.1)

        self.optimizer = optim.AdamW(self.model.parameters(), lr=self.lr, weight_decay=0.05)

        # Set up learning rate scheduler: warmup followed by cosine annealing
        steps_per_epoch = len(self.train_loader)
        total_steps = self.epochs * steps_per_epoch
        warmup_steps = 5 * steps_per_epoch
        decay_steps = total_steps - warmup_steps
        
        warmup_scheduler = torch.optim.lr_scheduler.LinearLR(
            self.optimizer, start_factor=0.1, end_factor=1.0, total_iters=warmup_steps
        )
        cosine_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            self.optimizer, T_max=decay_steps, eta_min=0.001 * self.lr
        )

        self.scheduler = torch.optim.lr_scheduler.SequentialLR(
            self.optimizer, schedulers=[warmup_scheduler, cosine_scheduler], milestones=[warmup_steps]
        )

        self.model, self.optimizer, self.train_loader, self.val_loader, self.scheduler = (
            self.accelerator.prepare(
                self.model, self.optimizer, self.train_loader, self.val_loader, self.scheduler
            )
        )

        # Load prev training state if checkpoint_dir is provided
        self.start_epoch = 0
        self.training_history = {'val_loss': [], 'val_acc': [], 'val_top5': [], 'train_loss': []}

        if checkpoint_dir:
            self.accelerator.load_state(checkpoint_dir)
            # Load training metadata (epoch number, best metrics, history, etc.)
            metadata_path = os.path.join(checkpoint_dir, 'training_metadata.json')
            if os.path.exists(metadata_path):
                with open(metadata_path, 'r') as f:
                    metadata = json.load(f)
                    self.start_epoch = metadata.get('epoch', 0) + 1  # Resume from next epoch
                    self.best_val_loss = metadata.get('best_val_loss', float('inf'))
                    self.best_val_acc = metadata.get('best_val_acc', 0.0)
                    self.best_val_top5 = metadata.get('best_val_top5', 0.0)
                    self.training_history = metadata.get(
                        'history',
                        {'val_loss': [], 'val_acc': [], 'val_top5': [], 'train_loss': []}
                    )
                    self.training_history.setdefault('val_top5', [])
                    print(f"Loaded training state from {checkpoint_dir} (resuming from epoch {self.start_epoch})")
            else:
                print(f"Loaded training state from {checkpoint_dir} (no metadata found, starting from epoch 0)")
                self.best_val_loss = float('inf')
                self.best_val_acc = 0.0
                self.best_val_top5 = 0.0
        else:
            self.best_val_loss = float('inf')
            self.best_val_acc = 0.0
            self.best_val_top5 = 0.0

        self.best_test_loss = float('inf')
        self.best_test_acc = 0.0

    def save_training_log(self, epoch, train_loss, val_loss, val_acc, val_top5):
        """Save training history and metadata every epoch"""
        os.makedirs(self.save_dir, exist_ok=True)

        # Update training history
        self.training_history['train_loss'].append(train_loss)
        self.training_history['val_loss'].append(val_loss)
        self.training_history['val_acc'].append(val_acc)
        self.training_history['val_top5'].append(val_top5)

        # Save full training metadata (always updated)
        metadata = {
            'epoch': epoch,
            'best_val_loss': self.best_val_loss,
            'best_val_acc': self.best_val_acc,
            'best_val_top5': self.best_val_top5,
            'history': self.training_history
        }
        metadata_path = os.path.join(self.save_dir, 'training_metadata.json')
        with open(metadata_path, 'w') as f:
            json.dump(metadata, f, indent=2)

    def save_model_if_best(self, val_loss, val_acc=0.0, val_top5=0.0):
        """Save model checkpoint only if it's the best so far"""
        if val_loss < self.best_val_loss:
            self.best_val_loss = val_loss
            self.best_val_acc = val_acc
            self.best_val_top5 = val_top5
            os.makedirs(self.save_dir, exist_ok=True)

            # save train state
            self.accelerator.save_state(
                os.path.join(self.save_dir)
            )

    def run_epoch(self, epoch):
        self.model.train()
        running_loss = 0.0
        total = 0

        pbar = tqdm(self.train_loader, desc=f"Epoch {epoch+1}/{self.epochs} - Training", leave=False)
        for i, (images, labels) in enumerate(pbar):
            images, labels = images.to(self.device), labels.to(self.device)

            self.optimizer.zero_grad()

            with self.accelerator.autocast():
                outputs = self.model(images)

                loss = self.criterion(outputs, labels)
                self.accelerator.backward(loss)
                if self.accelerator.sync_gradients:
                    self.accelerator.clip_grad_norm_(
                        self.model.parameters(), max_norm=1.0
                    )

            self.optimizer.step()
            self.scheduler.step()

            running_loss += loss.item() * images.size(0)
            total += labels.size(0)
            pbar.set_postfix({'loss': loss.item(), 'lr': self.scheduler.get_last_lr()[0]})

            if i % self.log_every == 0:
                if self.wandb_logging:
                    wandb.log({
                        'train_loss': loss.item(),
                        'lr': self.scheduler.get_last_lr()[0],
                    })
            
            # reduce gpu power
            time.sleep(0.1)

        avg_loss = running_loss / total
        return avg_loss

    def run_validation(self):
        self.model.eval()
        running_loss = 0.0
        correct_top1 = 0
        correct_top5 = 0
        total = 0

        pbar = tqdm(self.val_loader, desc="Validation", leave=False)
        with torch.inference_mode():
            for images, labels in pbar:
                images, labels = images.to(self.device), labels.to(self.device).long()

                outputs = self.model(images)
                loss = self.criterion(outputs, labels) 
                running_loss += loss.item() * images.size(0)

                # Compute top-1 and top-5 accuracy
                _, predicted = torch.max(outputs, 1)
                correct_top1 += (predicted == labels).sum().item()

                top5 = torch.topk(outputs, k=5, dim=1).indices
                correct_top5 += (top5 == labels.unsqueeze(1)).any(dim=1).sum().item()

                total += labels.size(0)

                pbar.set_postfix({'loss': loss.item()})
        
        avg_loss = running_loss / total
        top1_accuracy = correct_top1 / total
        top5_accuracy = correct_top5 / total
        return avg_loss, top1_accuracy, top5_accuracy

    def run_training(self):
        for epoch in tqdm(range(self.start_epoch, self.epochs), desc="Epochs", unit="epoch"):
            train_loss = self.run_epoch(epoch)
            print(f"Training - Epoch {epoch + 1}, Loss: {train_loss:.4f}")

            val_loss, val_acc, val_top5 = self.run_validation()
            print(f"Validation - Acc@1: {val_acc:.4f}, Acc@5: {val_top5:.4f}, Loss: {val_loss:.4f}")

            self.best_val_acc = max(self.best_val_acc, val_acc)
            self.best_val_top5 = max(self.best_val_top5, val_top5)

            # Save model checkpoint only if best (updates best_val_loss and best_val_acc)
            self.save_model_if_best(val_loss, val_acc, val_top5)

            # Save training log every epoch (after updating best metrics)
            self.save_training_log(epoch, train_loss, val_loss, val_acc, val_top5)

            if self.wandb_logging:
                wandb.log({
                    # 'train_loss': train_loss,
                    'val_loss': val_loss,
                    'val_acc': val_acc,
                    'val_top5': val_top5,
                    'lr': self.scheduler.get_last_lr()[0],
                    'epoch': epoch
                })

            if self.early_stopper(val_loss):
                break

        print(f"Training complete. Best validation metrics: Loss: {self.best_val_loss:.4f}, Acc: {self.best_val_acc:.4f}")


# Example usage:
if __name__ == '__main__':
    trainer = Trainer(
        epochs=400,
        batch_size=768,
        lr=0.0015,
        save_dir='./convmlpnet-hybrid2-b768-lr0.0015-decay0.001',
        wandb_logging=True,  # set to True if you wish to log with wandb
        wandb_name='convmlpnet-hybrid2-b768-lr0.0015-decay0.001', # wandb name
    )
    trainer.run_training()
