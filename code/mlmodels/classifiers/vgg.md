## VGG

VGG is one the architecture which got fame in ImageNet competition in 2012 and while it didn't win it truly got loved by ml community. Original paper is available here: [Very Deep Convolutional Networks for Large-Scale Image Recognition](https://arxiv.org/abs/1409.1556)  

VGG heavily uses Convolution Layer and proves that if we stuck enough of them on top of each other it can deliver exceptional results by learning a very deep representation of different features. This paper recommends following configurations of their main architecture: VGG 11, VGG 13, VGG 16, and VGG 19 which is also summarized in table below:

![alt text](resources/vgg_architecure_configuration.png "Arch")

Because those architectures basically go only depper we gonna implement a class `VGG` which based on confiration from a dictionary: `VGG_types` will build as many layers as necessary. The same approach we gonna use with 3D implementation of VGG called `LunaVGG`

