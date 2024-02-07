# Introduction
## Efficient Net
The previous architectures I implemented were getting more and more complex (in terms of number of parameters) therefore I was waiting to get my hands on EfficientNet which is well known and loved architecture which achieved great results on Imagenet with much less parameters. 

EfficientNet is coming from this paper: [Rethinking Model Scaling for Convolutional Neural Networks)](https://arxiv.org/abs/1905.11946) where authors:
> Systematically study model scaling and identify that carefully balancing network depth, width, and resolution can lead to better performance.

I believe that the idea for this paper come from the fact that different researchers have achieved great results when scaling up their base architecture in one of the dimensions: depth, width, and image size. However the process has been moreless random or heavily manually tuned. Therefore they focused on question: is there a principled method to scale up ConvNets that can achieve better accuracy and efficiency?


## ConvNet scaling
The central part of this research is **compound scaling method** which uniformly scales network width, depth and resolution with a set of fixed scaling coefficients.
> For example, if we want to use 2N times more computational resources, then we can simply increase the network depth by αN, width by βN, and image size by γN, where α,β,γ are constant coefficients determined by a small grid search on the original small model.

Simple comparison of different scaling is here:    
![alt text](resources/efficientnet_scaling.png "Arch")  

> Intuitively, the compound scaling method makes sense because if the input image is bigger, then the network needs more layers to increase the receptive field and more channels to capture more fine-grained patterns on the bigger image.

The authors proposed a better way of deciding of depth, width and resolution of the entwork without emperical testing of different architectures. While compound scaling method means that authors don't need a new architecture per se just proving that using existing one like ResNet or MobileNet as baseline and then scale it in 3 dimensions will give a better result than e.g. ResNet152 which was biggest and winning solution at that time. So basically authors are saying give me building blocks and I will build you optimized network. However, authors also proposed a new kind-of-mobile architecture called EfficientNet. 

Finding the right set of depth, width, resolution creates a huge design space so to reduce it they restrict that all layers must be scaled uniformly with constant ratio. Scaling in each dimension has it's own issues:
1. **Depth** - very deep networks are hard to train due to dimishing gradient problem (solved by: skip connections and batch normalization). Still ResNet1000 has a simmilar accuracy to ResNet100
2. **Width** - very wide shallow networks are easy to train and gives fastly good results but because they don't produce higher-lever of features (deep in netowrk) they usually cannot achieve sota accuracy
3. **Resolution** - Empirical research shows that using 299x299 or even 480x480 (GPipe) image size gives better result but the accuracy gain fastly saturates. 

However of course we cannot increase all of those dimensions at inifnite level to achieve best accuracy because of computational limit. Therefore they proposed compound scaling method, which use a compound coefficient φ to uniformly scales network width, depth, and resolution in a principled way:

![alt text](resources/efficientnet_compund_logic.png "Arch")   
where α, β, γ are constants that can be determined by a small grid search.

## EfficientNet Baseline
Authors were using multi-objective neural architecture search for findinig the best base architecture. As a result EfficinetB0 was created with following structure:

![alt text](resources/efficientnet_b0_arch.png "Arch")   

where MBConv is combination of two building blocks:
1. MobileNet inverted bottleneck (inc. depthwise and pointwise convolution)
1. SENET squeeze-and-excitation optimisation  
however, the paper doesn't specify exactly where the SE block goes as it's flexible therefore I used the same implementation as in torchvision.

# Implementation
Because our data is 3D I used following 3D blocks:
- `Conv3d`
- `BatchNorm3d`
- `AvgPool3d`

and two optim algorithms were tested:
- SGD()
- Adam()

for sample dataset of 64 image (in shuffled batches of 32) the process to converging took 10h and 250 epochs with SGD:

![alt text](resources/efficientnet_loss.png "Arch")

![alt text](resources/efficientnet_accuracy.png "Arch")


