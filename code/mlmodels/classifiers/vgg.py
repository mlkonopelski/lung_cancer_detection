import torch
import torch.nn as nn
from torchinfo import summary

VGG_types = {
    'VGG11': [64, 'M', 128, 'M', 256, 256, 'M', 512, 512, 'M', 512, 512, 'M'],
    'VGG13': [64, 64, 'M', 128, 128, 'M', 256, 256, 'M', 512, 512, 'M', 512, 512, 'M' ],
    'VGG16': [64, 64, 'M', 128, 128, 'M', 256, 256, 256, 'M', 512, 512, 512, 'M', 512, 512, 512, 'M',],
    'VGG19': [64,64,'M',128,128,'M',256,256,256,256,'M',512,512,512,512,'M',512,512,512,512,'M',],
    }

class VGG(nn.Module):
    def __init__(self, 
                 architecture: str,
                 in_channels: int = 3,
                 num_classes: int = 1000,
                 img_height: int = 224,
                 img_width: int = 224,
                 num_linear_neurons: int = 4096) -> None:
        super().__init__()
    
        self.architecture = VGG_types[architecture]
        self.in_channels = in_channels
        self.num_classes = num_classes
        self.img_height = img_height
        self.img_width = img_width
        self.num_linear_neurons = num_linear_neurons
        
        self.conv_layers = self.build_conv_layers()
        self.flatten = nn.Flatten()
        self.fc_layers = self.build_fc_layers()
    
    def build_fc_layers(self):
        
        factor = 2 ** self.architecture.count('M')
        img_resized = (self.img_height // factor, self.img_width // factor )
        last_cnn_features = next(x for x in self.architecture[::-1] if type(x) == int)
        
        return \
        nn.Sequential(
            nn.Linear(last_cnn_features * img_resized[0] * img_resized[1], self.num_linear_neurons),
            nn.ReLU(),
            nn.Dropout(),
            nn.Linear(self.num_linear_neurons, self.num_linear_neurons),
            nn.ReLU(),
            nn.Dropout(),
            nn.Linear(self.num_linear_neurons, self.num_classes),
            nn.Softmax()
        )
    
    def build_conv_layers(self):
        
        layers = []
        in_features = self.in_channels
        
        for config in self.architecture:
            if config == 'M':
                layers.append(nn.MaxPool2d(kernel_size=2, stride=2))
            else:
                out_features = config
                conv_layer = [nn.Conv2d(in_channels=in_features, out_channels=out_features, kernel_size=3, stride=1, padding=1),
                              nn.ReLU()]
                layers.extend(conv_layer)
                in_features = out_features
        
        return nn.Sequential(*layers)
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out = self.conv_layers(x)
        out = self.flatten(out)
        out = self.fc_layers(out)
        return out


class LunaVGG(VGG):
    def __init__(self, 
                 architecture: str, 
                 in_channels: int = 1, 
                 num_classes: int = 2, 
                 img_depth: int = 48, 
                 img_height: int = 32, 
                 img_width: int = 32,
                 num_linear_neurons: int = 4096) -> None:
        self.img_depth = img_depth
        super().__init__(architecture, in_channels, num_classes, img_height, img_width, num_linear_neurons)
        
           
    def build_fc_layers(self):
        
        factor = 2 ** self.architecture.count('M')
        img_resized = (self.img_depth // factor, self.img_height // factor, self.img_width // factor )
        last_cnn_features = next(x for x in self.architecture[::-1] if type(x) == int)
        
        return \
        nn.Sequential(
            nn.Linear(last_cnn_features * img_resized[0] * img_resized[1] * img_resized[2], self.num_linear_neurons),
            nn.ReLU(),
            nn.Dropout(),
            nn.Linear(self.num_linear_neurons, self.num_linear_neurons),
            nn.ReLU(),
            nn.Dropout(),
            nn.Linear(self.num_linear_neurons, self.num_classes),
            nn.Softmax()
        )
    
    def build_conv_layers(self):
        
        layers = []
        in_features = self.in_channels
        
        for config in self.architecture:
            if config == 'M':
                layers.append(nn.MaxPool3d(kernel_size=2, stride=2))
            else:
                out_features = config
                conv_layer = [nn.Conv3d(in_channels=in_features, out_channels=out_features, kernel_size=3, stride=1, padding=1),
                              nn.ReLU()]
                layers.extend(conv_layer)
                in_features = out_features
        
        return nn.Sequential(*layers)


if __name__ == '__main__':

    vvg19 = VGG(architecture='VGG19')
    summary(vvg19, input_size=(1, 3, 224, 224))

    luna_vvg16 = LunaVGG(architecture='VGG11')
    summary(luna_vvg16, input_size=(1, 1, 32, 48, 48))
    