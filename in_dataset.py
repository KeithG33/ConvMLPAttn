from datasets import load_dataset
import torch


class inDataset(torch.utils.data.Dataset):
    def __init__(self, dataset, train_transforms=None, val_transforms=None):
        # Load the dataset
        self.dataset = load_dataset("imagenet-1k", split=f"{dataset}")
        self.split = dataset

        # Define the transformations
        self.train_transforms = train_transforms
        self.val_transforms = val_transforms


    def __getitem__(self, idx):
        # Sample row idx from the loaded dataset
        sample = self.dataset[idx]

        # Split up the sample example into an image and label variable
        data, label = sample['image'], sample['label']

        # Ensure data RGB
        data = data.convert('RGB')

        # Apply transformations
        if self.split == 'train':
            data = self.train_transforms(data)
        elif self.split == 'validation':
            data = self.val_transforms(data)
        elif self.split == 'test':
            data = self.val_transforms(data)

        return data, label

    def __len__(self):
        return len(self.dataset)