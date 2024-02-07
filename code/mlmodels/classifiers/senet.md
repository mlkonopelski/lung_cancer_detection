# Squeeze-and-Excitation Networks
CNN are able to produce image representations that capture hierarchical patterns and attain global theoretical receptive fields. This image represenation is obtained by collection of filters expresses neighbourhood spatial connectivity patterns along input channels. Therefore models and research has focused on capturing only those properties of an image that are most salient for a given task by redesigning the size and quantity of filters as well stacking features. However the authors of: [Squeeze-and-Excitation Networks](https://arxiv.org/abs/1709.01507) has designed a new architectural block with a goal of:

> improving the quality of representations produced by a network by explicitly modelling the interdependencies between the channels of its convolutional features. To this end, we propose a mechanism that allows the network to perform feature recalibration, through which it can learn to use global information to selectively emphasise informative features and suppress less useful ones.

![alt text](resources/senet-architecture.png "Arch")

The SE block consists of 3 operations:
1. **Squeeze** operation, which produces a channel descriptor by aggregating feature maps across their spatial dimensions (H × W )
1. **Excitation** takes the embedding as input and pro- duces a collection of per-channel modulation weights.
1. Applying weights to original input. 


>  The role it performs at different depths differs throughout the network. In earlier layers, it excites informative features in a class-agnostic manner, strengthening the shared low-level representations. In later layers, the SE blocks become increasingly specialised, and respond to different inputs in a highly class-specific manner

SE block has benefit of being very flexible. It can both be applied after each convolution+activation or around entire architecturall block here around SE on Inception or used on residual block on ResNet:
![alt text](resources/senet-blocks.png "Arch")

