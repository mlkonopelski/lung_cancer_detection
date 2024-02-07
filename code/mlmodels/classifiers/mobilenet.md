# MobileNets
## Idea
Research in previous years has focused on deeper and wider CNN in order to achieve better performance in competitions like ImageNet. However those architectures started to be more academic especially for real-world applications like: robotics, automotive, mobile devices etc. 
This paper: [MobileNets: Efficient Convolutional Neural Networks for Mobile Vision Applications](https://arxiv.org/pdf/1704.04861.pdf)
> describes an efficient network architecture and a set of two hyper-parameters in order to build very small, low latency models that can be easily matched to the design requirements for mobile and embedded vision ap- plications.


## Mobile Block
Mobile Nets are using factorized convolutions in two forms:
- depthwise convolution
- pointwise convolution

![alt text](resources/mobilenet_factorized_conv.png "Arch") 

> A standard convolution both filters and combines inputs into a new set of outputs in one step. The depthwise separable convolution splits this into two layers, a separate layer for filtering and a separate layer for combining. This factorization has the effect of drastically reducing computation and model size.

So instead of using traditional computing cost of:
$$D{K} · D{K} · M · N ·D{F} ·D{F} $$


this approach have:
$$D{K} · D{K} · M · D{F} · D{F} + M · N · D{F} · D{F}$$

,where D{K} - kernel resolution, D{F} - input resolution, M - input features, N - output features

authors argue that this method is 8/9 times faster without losing significantly accuracy. 

## Implementation
The detailed architecture:

![alt text](resources/mobilenet_architecture.png "Arch")

> All layers are followed by a batchnorm and ReLU nonlinearity with the exception of the final fully connected layer which has no nonlinearity and feeds into a softmax layer for classification.

While pointwise conv is no brainer, depthwise requiers a bit of thining because conv layers are not fully connected. So I created my own solution (which is terrible unoptimized becuase of loop). However after some research I realized that `Conv2D` has a parameter `group` which when `== in_channels` does exactly this job.

**Note:** The authors also introduced concept of: resolution multiplier however I decided to not implement it as the point simple to reduce the size of imput image and therefore reduce the number of parameters. Their experiment showed that using 25% smaller pictures reduces accuracy by 2% (70.6% -> 68.4%)
{: .note}

# MobileNets V2
The novelty in this architecture is a new lightweight module: an inverted residual structure where the shortcut connections are between the thin bottleneck layers
> This module takes as an input a low-dimensional compressed representation which is first expanded to high dimension and filtered with a lightweight depthwise convolution. Features are subsequently projected back to a lowdimensional representation with a linear convolution. 

## Building blocks
There are 3 basic terms:
1. Depthwise Separable Convolutions - as in V1
1. Linear Bottlenecs - adding additional 1x1 conv layer with x6 more out feature maps than in. It's called expansion.
1. Inverted Residuals - residual connection inside bottleneck block if dimensions match

The summary of them is simple showed here:
![alt text](resources/mobilenetV2_residual_bottleneck.png "Arch")

while in depth view to Bottleneck block:
![alt text](resources/mobilenetV2_bottleneck_details.png "Arch")


## Architecture
Fortunately this time the architecture is well structured so it's easy to create those blocks and layers in a loop:

![alt text](resources/mobilenetV2_architecture.png "Arch")

Additionally there is a 50% Dropout before fully connected layer. Also authors proposed to use ReLU6 for activations since they are more suited for lower precision numbers. 

