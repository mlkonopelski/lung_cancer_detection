## GoogLeNet a'ka Inception

While the codename Inception is taken from a meme this architecture is no joke ;)  
I decided to use it a my next implementation because in ImageNet war it not only won (in 2014) but also proposed a new architecture. Moreover, every paper Christian Szegedy authors it's an interesting read and they try to support every creative approach with intuition, math or research behind it. 

The new approach was to instead of using sequential convolutional layers they decided to build parallel convolution operation which they called inception. 

> The main idea of the Inception architecture is based on finding how an optimal local sparse structure in a convolutional vision network can be approximated and covered by readily available dense components

This inception module is presented below and you might think about it as network within network (that's why inception)

![alt text](resources/inception_module.png "Arch")

This module is repeated multiple time is sequential manner together with Max pooling to reduce dimitionality. The important detail is however that implementing straight forward such as block would result is massive amount of parameters if e.g. imput is 28x28x512


## Versions

### Inception V1
This it the original design that won ImageNet 2014: [Going Deeper with Convolutions](https://arxiv.org/abs/1409.4842)

One of the most interesting features of this model is using 3 Softmax output layers inside the network and weight it for loss function. The reasoning for that is:
> By adding auxiliary classifiers connected to there intermiediete layers, we would expect to encourage discrimination in the lower stages in the classifier, increase the gradient signal that gets propagated back, and provide additional regularization. 

The while model consists of 22 layers (inlcuding 9 inceptions layers and fully connected before SoftMax

Additional implemetations details:
1. Activation function after each convolution: **ReLU**
1. Generalization with Dropout laters: **70%**
1. Keeping original image size therefore each module has appropriate padding based on kernel size: 3x3 -> 1; 5x5-> 2
1. training:
    - optim: SGD with 0.9 Mommentum
    - lr: scheduled with 4% deccrease every epoch
    - ensemble: of 7 same models but trained on different samples

Full architecture: 
![alt text](resources/inception-architecture.png "Arch")

while the number of features of each block and module: 
![alt text](resources/inception-features.png "Arch")


### Inception V2 (& V3)
Those are simmilar designs while I believe that V2 is one of the biggest breakthroughts in ML because it includes `BatchNormalization`. Research for this design can be found in those two papers:  
1. [Batch Normalization: Accelerating Deep Network Training by Reducing Internal Covariate Shift](https://arxiv.org/abs/1502.03167)
1. [Rethinking the Inception Architecture for Computer Vision](https://arxiv.org/abs/1512.00567)

> Batch Normalization allows us to use much higher learning rates and be less careful about initialization. It also acts as a regulatizer, in some cases eliminating the need for Dropout. Applied to a state-of-the-art image classification model, Batch Normalization achieves the same accuracy with 14 times fewer training steps, and beats the original model by a significant margin. 

### Inception V4
[ ] Inception V4 to be implemented after ResNet


### LunaInception
Is build based on Inception V4 and includes 3D Convolutions instead od 2D. 