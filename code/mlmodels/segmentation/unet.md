# Summary
The main idea is to use Convolution operations to extract features out of image simmilar to classicial CNN for classification but with additional Decoder/Expander part which will be consisted of CNN Upsampling blocks to increase dimensions of feature map up to original size.  
This way would be possible to both learn spatial representation of image and be consistent with input/output resolution. 

## Original UNET
The original architecture looks like this:
![alt text](resources/unet-custom-img.tiff "classic-unet-architecture")


which has following interesting parts:  
1. Convolutional Layers with params:
    * `kernel_size=(3,3)`
    * `stride=(1,1)`
    * `padding=0`
1. Max Pooling with params:
    * `kernel_size=(2,2)`
    * `stride=(2,2)`
1. Upsampling with params:
    * `kernel_size=(2,2)`
    * `stride=(2,2)`
    * `paddng=0`
1. Croping image to target size
1. Concat original feature map with after upsampling map



## UNET for LUNA
There are multiple changes to original UNET architecture in order to better fit to our problem domain:  
1. We want to have the same output size of network as input herefore each convolution block will have `padding=1` 
1. We output 1-dim array with values <0, 1> therefore we add `Sigmoid` layer at the end
1. Add the beggining each input array gets normalized for better learning of network. 


## Tricks to make it work
1. We want use full 512x512 mask as input for training because this dataset is highly imbalanced. 
True values consits only little percentage of all cells of interested. 
Therefore model will be trained on 64x64 mini crops around nodule as UNET is flexible on dimensions and 
learn mostly weghts on feature maps. Also this way we'are goinf to add naturally augmentation. 
1. 