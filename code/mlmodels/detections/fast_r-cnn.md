# Introduction
The fast region convolutional network (fast r-cnn) has been introduced to sort the biggest problem with original architecture: [Rich feature hierarchies for accurate object detection and semantic segmentation](https://arxiv.org/abs/1311.2524) which was slow processing speed due to processing each (out of 2000) potential region through deep cnn. The solution was to process the whole image and run per-region processing only through very shallow neural network. This paper also added a new cnn component which was "ROI pooling" which is bascially smart (differentiable) way to fitting any scale and ratio bounding box feature map to fixed hxw scale. 

However this wasn't my favourite paper to implement because of difficulty in understanding key components of architecture and the consequence of it is that I saw multiple articles on medium/youtube etc which had a different understanding of them and I found them all wrong (no one has tried to implement it in pure torch). There is an authors python implementation [github/rbgirschick](https://github.com/rbgirshick/fast-rcnn/tree/master/lib) but it's using Caffe which I'm not familiar with (so it needs couple of days to debug it line by line)

# Architecture
![alt text](resources/fast-rcnn-arch.png "Arch")  

## backbone
As usually any network can be backbone for object detection but authors experimented and pointed that vgg16 gives best relult in size/accuracy. 
![alt text](resources/fast-rcnn-vgg16-arch.png "Arch")  

The paper states:
> The network first processes the whole image with several convolutional (conv) and max pooling layers to produce a conv feature map.


## roi pooling

> First, the last max pooling layer is replaced by a RoI pooling layer that is configured by setting H and W to be compatible with the net’s first fully connected layer (e.g., H = W = 7 for VGG16).

The paper is using last conv layer as input to roi pooling which is 14x14 in case of vgg16. I do however think that's in case of small objects in picture (which takes < 5% of w or h) is more reasonable to use feature map of higher dimension e.g. 128x128. My trouble with this implementation is that if selective search is giving me 5 possible bounding boxes 3 of them have iou > 50% and are labeled as 1 and other 2 as 0. If I do roi pooling on 14x14 feature map eac of those roi examples will have the same output of roi pooling but different target. On the other hand we could project the roi on the 1st layer of vgg and therefore solve this issue - in this case why would I use all others cnn layers. 


## fc output
The results of roi pooling are fed to two fully connected layers:
1. one-hot vector with classes for softmax probabilities
2. per-class bounding-box regression offsets

# Training
## mini-batch training
> We propose a more efficient training method that takes
advantage of feature sharing during training. In Fast RCNN training, stochastic gradient descent (SGD) minibatches are sampled hierarchically, first by sampling N images and then by sampling R/N RoIs from each image.
Critically, RoIs from the same image share computation
and memory in the forward and backward passes. Making
N small decreases mini-batch computation. For example,
when using N = 2 and R = 128, the proposed training
scheme is roughly 64× faster than sampling one RoI from
128 different images (i.e., the R-CNN and SPPnet strategy)


## loss function
The loss is calculated speratly for class prediction and bbox prediction (only for object class) and added together. 
![alt text](resources/fast-rcnn-loss1.png "Arch")  
![alt text](resources/fast-rcnn-loss2.png "Arch")  
