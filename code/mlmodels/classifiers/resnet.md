The recent advancements in Computer Vision (prior to publishing this paper) were due to increasing the depth of Convolutional Neural Networks and therefore increasing depth of representations. However, it doesn’t come without cost in form of training difficulties. 


> Deep networks naturally integrate low/mid/high- level features and classifiers in an end-to-end multi- layer fashion, and the “levels” of features can be enriched by the number of stacked layers (depth). 

The first problem of Vanishing/Exploding gradients has been initially addressed by few innovations as: normalised initialisation and intermediate normalisation layers. The second problem which occur was “degradation” of accuracy with bigger depth, which wasn’t caused by overfitting (training error was increasing). 

> Instead of hoping each few stacked layers directly fit a desired underlying mapping, we explicitly let these layers fit a residual mapping. (…) We hypothesize that it is easier to optimize the residual mapping than to optimize the original, unreferenced mapping. To the extreme, if an identity mapping were optimal, it would be easier to push the residual to zero than to fit an identity mapping by a stack of nonlinear layers. 

The residual mapping was obtained by adding “shortcut connections” which in practice means that they perform identity mapping by adding outputs of one or few layers to input of them. This can be showed as:

![alt text](resources/resnet-features-block.png "Arch")

The architecture of Residual Network is simmilar to traditional sequentional CNN (like VGG) but every each CNN block has added "shortcut connections". The top of the network is showed here:

![alt text](resources/resnet-network.png "Arch")

I above we can see a problem an issue marked as dotted line when feature map of depth 64 is added to feature map of depth 128 and lower image size. Authors proposed two solutions:
> (A) The shortcut still performs identity mapping, with extra zero entries padded for increasing dimensions. This option introduces no extra parameter; (B) The projection shortcut in Eqn.(2) is used to match dimensions (done by 1×1 convolutions).

When researchers were experimenting with deeper res nets they modified also the building block of "shortcut connection" due to bottleneck problem of increased computational needs. Instead of using projections which were expensive they used 3 stacked conv layers: 1x1, 3x3, 1x1 where 1x1 were introduced to reduce and then increase (restore) dimensions. The comparison is showed below:
![alt text](resources/resnet-features-block2.png "Arch")

I decided to implement two versions of this network: ResNet34 and ResNet152 since they differ in residual building blocks. Fortunately authors have included detailed explanation of each archtiecture:

![alt text](resources/resnet-network-details.png "Arch")

The biggest challange in this implementation was to create flexible building blocks because each layer cosists of blocks of different number of feature maps and different input/output sizes. Also "" sometimes requiered downsampling or increasing depth (feature maps). This was done either by:
- ResNet32:
    - `PadDownSampler` - No parameter requiered by using max pooling + zero padding
    - `ResNetBasicBlock` - Two 3x3 convolution blocks
- ResNet152:
    - `ConvDownSampler` - Mapping learned by 1x1 convolution and stride = 2
    - `ResNetBottleneckBlock` - 1x1 followed by 3x3 and 1x1 convolution blocks   

**Samplers are necassary to match shape of input to residual layer to it's output in order to create shortcut connection**  
To make it work `ResNet` has helper method: `_make_layer()` which have following logic (naming convention based on diagram above and resnet152):
1. layer: conv2_1:
    - **sampler**: do NOT decrease size + increase features to 256 to match output 
    - **residual**: do NOT decrease size
1. layer: conv2_2 to conv2_3:
    - **sampler**: input shape to layer == output so sampler no needed
    - **residual**: do NOT decrease size
1. layer: conv3_1;
    - **sampler**: decrease size + increase features to 512 to match the output
    - **residual**: NOT decrease size by first convolution in block with stride=2
1. layer: conv3_2 to conv3_8 ...:
    - **sampler**: nput shape to layer == output so sampler to needed
    - **residual**: do NOT decrease size  
(...) 



