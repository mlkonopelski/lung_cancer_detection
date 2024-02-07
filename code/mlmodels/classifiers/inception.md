## GoogLeNet a'ka Inception

While the codename Inception is taken from a meme this architecture is no joke ;)  
I decided to use it a my next implementation because in ImageNet war it not only won (in 2014) but also proposed a new architecture. Moreover, every paper Christian Szegedy authors it's an interesting read and they try to support every creative approach with intuition, math or research behind it. 

The new approach was to instead of using sequential convolutional layers they decided to build parallel convolution operation which they called inception. 

> The main idea of the Inception architecture is based on finding how an optimal local sparse structure in a convolutional vision network can be approximated and covered by readily available dense components

This inception module is presented below and you might think about it as network within network (that's why inception)

![alt text](resources/inception_module.png "Arch")

This module is repeated multiple time is sequential manner together with Max pooling to reduce dimitionality. The important detail is however that implementing straight forward such as block would result is massive amount of parameters if e.g. imput is 28x28x512


## Inception V1
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


## Inception V2 (& V3)
Those are simmilar designs while I believe that V2 is one of the biggest breakthroughts in ML because it includes `BatchNormalization`. Research for this design can be found in those two papers:  
1. [Batch Normalization: Accelerating Deep Network Training by Reducing Internal Covariate Shift](https://arxiv.org/abs/1502.03167)
1. [Rethinking the Inception Architecture for Computer Vision](https://arxiv.org/abs/1512.00567)


However BN is not the only change. Actually the architecture changed a lot and while Paper doesn't give pretty image of new architecture I will try to go step by step. 

### Batch Normalization
> Batch Normalization allows us to use much higher learning rates and be less careful about initialization. It also acts as a regulatizer, in some cases eliminating the need for Dropout. Applied to a state-of-the-art image classification model, Batch Normalization achieves the same accuracy with 14 times fewer training steps, and beats the original model by a significant margin. 

I our Model it was implemented after each Convolutional Layer and therefore we built `InceptionV2ConvBlock` which consists of 3 operations:
1. Conv 2D
1. Batch Normalization 
1. RELU  
And it's going to be base for all convolution operations in this architecture. 

### Inception Block
In contrast to V1 this time each Inception section will be different. Authors emphasize that previous amazing result of V1 was due to smart dimentionality reduction of Convolution layers. 
> Much of the original gains of the GoogLeNet network arise from a very generous use of dimension re- duction. This can be viewed as a special case of factorizing convolutions in a computationally efficient manner. Con- sider for example the case of a 1 × 1 convolutional layer followed by a 3 × 3 convolutional layer. In a vision network, it is expected that the outputs of near-by activations are highly correlated. Therefore, we can expect that their activations can be reduced before aggregation and that this should result in similarly expressive local representations.

Based on this principle 2 new approaches were introduced and implemented as inception blocks:
1. Factorization into smaller convolutions
1. Spatial Factorization into Asymmetric Convolutions (block 2 & 3)

### Inception Block 1

Teh first approach was simple. Instead of using 7x7 or 5x5 filters, it's better to use consectuive 3x3 ones. 

> Higher dimensional representations are easier to process locally within a network. Increasing the activations per tile in a convolutional network allows for more disentangled features. The resulting networks will train faster.

![alt text](resources/inceptionv2-feature1.png "Arch")

### Inception Block 2 & 3

This time innovation was to instead a large e.g. 7x7 filter use consectuive layers of 1x7 and 7x1 filters. This approach drastically decreased computional cost and I assume that gave good result. 

However experiemnts showed that:
> In practice, we have found that employing this factorization does not work well on early layers, but it gives very good results on medium grid-sizes (On m × m feature maps, where m ranges between 12 and 20). On that level, very good results can be achieved by using 1 × 7 convolutions followed by 7 × 1 convolutions.

![alt text](resources/inceptionv2-feature2.png "Arch")
![alt text](resources/inceptionv2-feature3.png "Arch")


### Reducing Size
Instead of using MaxPool or AvgPool for reducing size between Inception Blocks another subnet was created were inside operations using stride=2 were used to decrease size but increase depth of features in parallel. 

>Traditionally, convolutional networks used some pooling operation to decrease the grid size of the feature maps. In order to avoid a representational bottleneck, before applying maximum or average pooling the activation dimension of the network filters is expanded.

![alt text](resources/inceptionv2-feature5.png "Arch")

### New Auxulary output appeach
> Interestingly, we found that auxiliary classifiers did not result in improved convergence early in the training: the training progression of network with and without side head looks virtually identical before both models reach high accuracy. Near the end of training, the network with the auxiliary branches starts to overtake the accuracy of the network without any auxiliary branch and reaches a slightly higher plateau.

Therefore Authors proposed only one Auxulary output in the middle:  
![alt text](resources/inceptionv2-feature4.png "Arch")

arguing that:
> Instead, we argue that the auxiliary classifiers act as regularizer. This is supported by the fact that the main classifier of the network performs better if the side branch is batch-normalized [7] or has a dropout layer. This also gives a weak supporting evidence for the conjecture that batch normalization acts as a regularizer.


### Implementation challange
Unfortuantely this time authors didn't include the filter sizes for each convolution in Inception block stating that: 
> The detailed structure of the network, including the sizes of filter banks inside the Inception modules, is given in the supplementary material, given in the model.txt that is in the tar-file of this submission

However I couldn't find that plan so I had to derive them rougly based on V1 and knowing the input/output size of each block

### v2 vs V3
Inception V3 has the same architecture but different training specs:
1. RMSProp Optimizer
2. Factorized 7x7 convolutions
3. BatchNorm in the Auxillary Classifiers
4. Label Smoothing
5. Gradient Clipping with threshold 2.0

## Inception V4
[ ] Inception V4 to be implemented after ResNet


## LunaInception
[ ] Build based on Inception V4 and includes 3D Convolutions instead od 2D. 