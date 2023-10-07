## LeNet5

LeNet5 is classical CNN architecture proposed Yann LeCun in this paper: [Gradient Based Learning Applied to Do cument Recognition](http://vision.stanford.edu/cs598_spring07/papers/Lecun98.pdf)  


The architecture is simple (for nowadays models) and consists of 3 Convolution layers, 2 MaxPool layers and fully connected layers. To introduce non-linearity **Tanh** is used as activation function.  

![alt text](resources/lenet5_architecture.png "Arch")

What is important and what majority of implementations online forget is that feature maps in **C3 Layer** are not fully connected to all feature maps in **S2**. Instead Yann has provided a mapping and it's reflected `c3_mapping` is my code.  

![alt text](resources/lenet5_architecture_details.png "ArchDetails")

My LunaLeNet5 differs by:  
1. Adapting Conv and AvgPool layers to be 3D
1. changing out head to Linear(out=2) + Sigmoid() as we have a binary classification.  
1. Slight change to dimensions so now Model is natively available to process batches of Tensors  

---
TD:  
[ ] I haven't dig enough deep to check what kind of weight initialization was used for this model and left default pyTorch.