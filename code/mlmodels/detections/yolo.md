# Summary
The most popular model for real-time obejct detection. While it might not be perfect for this project as our objective is precision over speed. It's such a popular framework that I HAD TO use my hands on it.  

## algorithm

While the architecture has been chaning over the years the general idea has become the same. YOLO predicts mulitple bounding boxes in each of cell (image in divided into grid). Then an algorithm non-max surpression decides which bounding boxes are distinct enough to be final bounding boxes. Because the output is a matrix grid_size x grid_size x (count_bb * 5 + count_classes) it's a single step detection algoirthm (r-cnn are 2 steps) therefore it's so fast (hundrets of fps on modern gpu). The backbone of each yolo algorithm is usually CNN with mulitple layers and filters to learn the general features of the whole image.

# VERSIONS
## YOLOv1
paper: [You Only Look Once: Unified, Real-Time Object Detection](https://arxiv.org/abs/1506.02640) 

### Config
Output:
- grid_size = 7x7
- bounding boxes per cell = 2 
- 20 classes (orig: PASCAL VOC)

Architecture:  
![alt text](resources/yolov1-arch1.png "Arch")  
Additionally authors added 1x1 convolutions to reduce the number of feature maps and keep the number of parameters relatively low.

Loss:
![alt text](resources/yolov1-loss1.png "Arch")  


Training:  
Pre-train on ImageNet (input: 224x224) and resize for 448x448.


## YOLOv2
Paper = [YOLO9000:Better, Faster, Stronger](https://arxiv.org/pdf/1612.08242.pdf)
Best explanation: [ML For Nerds](https://youtu.be/PYpn1GSwWnc?si=KNbSMY83vhjo154b)

Key changes from YOLOv1:  
![alt text](resources/yolov2-newfeatures.png "Arch")  

1. batch norm - Batch Normalisation after each convolution layer instead of Dropout layer
2. hi-res classifier - train backbone classifier on ImageNet 448x448 instead of 224x224
3. convolutional - the whole network is based on convolutional layers (without fc at the head). This enables more output parameters and is flexible with input sizes (must be 32x)
4. new network - Instead of using VGG as backbone they use Darknet-19 (nothing fancy but smaller and more accurate)
5. dimension priors - before training all bounding boxes are fit with k-means (k=5) to find typical scale and ratio. Those 5 boes are used as anchor boxes
6. location prediction - instead of predicting w&h in reagrds to image 0,0 position w&h are calculated in regards to anchor boxes 0,0. Also now each bounding box has it's own onehot class vector
7. passthrough - Simmilar to resnet there is a skip connection from middle of network to deeper layer so initial information are not lost with decreasing size. In order to do that the bigger dimension image (dim x dim x depth) is divided into 4 and stacked on top of each other creating (dim/4 x dim/4 x depth*4)
8. hi-res detector - instead of grid cell 7 they do 13x13 and therefore are able to