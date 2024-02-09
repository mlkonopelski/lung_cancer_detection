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
