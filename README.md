# ConvMLPAttn

A hybrid vision architecture that combines convolutional blocks in early stages with MLP-based attention mechanisms in later stages. The model uses ConvNeXt V2 blocks for local feature extraction and a novel MLPAttn2D mechanism that generates attention matrices through an MLP, providing an alternative to standard dot-product attention that learns efficiently with low parameter networks.

Inspired by:
- [ConvNeXt V2: Co-designing and Scaling ConvNets with Masked Autoencoders](https://openreview.net/pdf?id=99XvUeDFYTD)
- [MLP-Attention: Improving Transformer Architecture with MLP Attention Weights](https://openreview.net/pdf?id=99XvUeDFYTD)

## Performance

\* Training and latency measurement all done on an RTX 3090

| Model | Params | Resolution | Top-1 (IN1k) | Latency* |
|-------|--------|------------|--------------|----------|
| ConvMLPAttn | 4.02M | 224 | 74.2% | 9.06ms |

## Architecture

The network consists of:
- Patch embedding with 4x downsampling
- Stage 1-2: ConvNeXt V2 blocks for efficient local feature learning
- Stage 3-4: MLP-generated attention blocks for global context modeling
- Hierarchical feature maps with progressive channel expansion
