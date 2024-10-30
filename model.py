import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.models import squeezenet1_1, inception_v3
from torchsummary import summary

import argparse

def get_argparser():
    parser = argparse.ArgumentParser(description="Define the CNN architecture for Food-101 dataset. You can run this to get a summary of the model.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument("--number_of_classes", type=int, default=101, help="Number of classes in the dataset")
    parser.add_argument("--image_input_share", type=tuple, default=(3, 512, 512), help="Input image shape (channels, height, width)")
    parser.add_argument("--model_name", type=str, default="inception_v3_ft", choices=["small", "simple", "inception_v3", "inception_v3_ft"], help="Name of the model to use")
    return parser


# Define the CNN architecture
class SmallConvNet(nn.Module):
    """
    Small ConvNet for Food-101 dataset
    """
    def __init__(self, num_classes=101):
        super(SmallConvNet, self).__init__()
        
        # Convolutional Block 1
        self.conv1 = nn.Conv2d(3, 32, kernel_size=3, stride=1, padding=1)
        self.drop1 = nn.Dropout(0.3)

        self.conv2 = nn.Conv2d(32, 32, kernel_size=3, stride=1, padding=1)
        
        # Convolutional Block 2 (Depthwise Separable Conv)
        self.conv3 = nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1, groups=32)  # Depthwise
        self.pointwise1 = nn.Conv2d(64, 64, kernel_size=1, stride=1)  # Pointwise\
        
        # Convolutional Block 3 (Depthwise Separable Conv)
        self.conv4 = nn.Conv2d(64, 128, kernel_size=3, stride=1, padding=1, groups=64)  # Depthwise
        self.pointwise2 = nn.Conv2d(128, 128, kernel_size=1, stride=1)  # Pointwise
        
        # Global Average Pooling layer
        self.global_avg_pool = nn.AdaptiveAvgPool2d((1, 1))

        self.drop2 = nn.Dropout(0.3)
        
        # Fully Connected Layer
        self.fc = nn.Linear(128, num_classes)
        
    def forward(self, x):
        # Block 1
        x = F.relu(self.conv1(x))
        x = self.drop1(x)
        x = F.relu(self.conv2(x))
        x = F.max_pool2d(x, kernel_size=2)
        
        # Block 2
        x = F.relu(self.pointwise1(self.conv3(x)))
        x = F.max_pool2d(x, kernel_size=2)
        
        # Block 3
        x = F.relu(self.pointwise2(self.conv4(x)))
        x = F.max_pool2d(x, kernel_size=2)
        
        # Global Average Pooling
        x = self.global_avg_pool(x)

        x = self.drop2(x)
        
        # Flatten the tensor
        x = torch.flatten(x, 1)
        
        # Fully Connected Layer
        x = self.fc(x)
        
        return x
    
class SimpleCNN(nn.Module):
    def __init__(self, num_classes):
        super(SimpleCNN, self).__init__()
        self.conv1 = nn.Conv2d(3, 32, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, padding=1)
        self.conv3 = nn.Conv2d(64, 128, kernel_size=3, padding=1)
        self.fc1 = nn.Linear(128 * 64 * 64, 256)
        self.fc2 = nn.Linear(256, num_classes)
        self.pool = nn.MaxPool2d(2, 2)
        self.dropout = nn.Dropout(0.5)
    
    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))  # 32x32 -> 16x16
        x = self.pool(F.relu(self.conv2(x)))  # 16x16 -> 8x8
        x = self.pool(F.relu(self.conv3(x)))  # 8x8 -> 4x4
        x = torch.flatten(x, 1)          # Flatten
        x = F.relu(self.fc1(x))
        x = self.dropout(x)
        x = self.fc2(x)
        return x
    

class DepthwiseSeparableConv(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=3, padding=1):
        super(DepthwiseSeparableConv, self).__init__()
        # Depthwise convolution
        self.depthwise = nn.Conv2d(in_channels, in_channels, kernel_size=kernel_size, padding=padding, groups=in_channels)
        # Pointwise convolution
        self.pointwise = nn.Conv2d(in_channels, out_channels, kernel_size=1)
    
    def forward(self, x):
        x = self.depthwise(x)
        x = self.pointwise(x)
        return x

class ResidualBlock(nn.Module):
    def __init__(self, in_channels, out_channels, downsample=False):
        super(ResidualBlock, self).__init__()
        self.downsample = downsample
        stride = 2 if downsample else 1
        
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1, stride=stride)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm2d(out_channels)
        
        self.skip_connection = nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=stride)
        self.bn_skip = nn.BatchNorm2d(out_channels)
    
    def forward(self, x):
        identity = x
        if self.downsample or x.shape[1] != self.conv1.out_channels:
            identity = self.bn_skip(self.skip_connection(x))
        
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out += identity  # Add skip connection (residual)
        return F.relu(out)
    
class Inception_v3(nn.Module):
    def __init__(self, num_classes):
        super(Inception_v3, self).__init__()
        self.model_ft = inception_v3(pretrained=True)

        for param in self.model_ft.parameters():
            param.requires_grad = False

        # Parameters of newly constructed modules have requires_grad=True by default
        # # Handle the auxilary net
        num_ftrs = self.model_ft.AuxLogits.fc.in_features
        self.model_ft.AuxLogits.fc = nn.Linear(num_ftrs, num_classes)
        # Handle the primary net
        num_ftrs = self.model_ft.fc.in_features
        self.model_ft.fc = nn.Linear(num_ftrs, num_classes)

    
    def forward(self, x):
        if self.training:
            # there is aux logits during training
            return self.model_ft(x)[0]
        else:
            return self.model_ft(x)
        

class Inception_v3_FT(nn.Module):
    def __init__(self, num_classes):
        super(Inception_v3_FT, self).__init__()
        self.model_ft = inception_v3(pretrained=True)

        for i, param in enumerate(self.model_ft.parameters()):
            if i < 150:
                param.requires_grad = False

    
    def forward(self, x):
        if self.training:
            # there is aux logits during training
            return self.model_ft(x)[0]
        else:
            return self.model_ft(x)

class ImprovedCNN(nn.Module):
    def __init__(self, num_classes):
        super(ImprovedCNN, self).__init__()
        
        # Initial Convolution Layer
        self.conv1 = nn.Conv2d(3, 16, kernel_size=3, padding=1)  # Input: 3x32x32 -> Output: 32x32x32
        
        # Residual Blocks with Depthwise Separable Convolutions
        self.resblock1 = ResidualBlock(16, 32, downsample=True)
        self.resblock2 = ResidualBlock(32, 64, downsample=True)
        
        # Depthwise Separable Convolutions
        self.conv2 = DepthwiseSeparableConv(64, 128)
        
        # Fully Connected Layers
        self.fc1 = nn.Linear(2097152, 128)
        self.fc2 = nn.Linear(128, num_classes)
        
        # Dropout
        self.dropout = nn.Dropout(0.5)
    
    def forward(self, x):
        # Initial Conv + BN + ReLU
        x = F.relu(self.conv1(x))  # Output: 32x32x32
        
        # Residual Blocks
        x = self.resblock1(x)  # Output: 64x16x16
        x = self.resblock2(x)  # Output: 128x8x8
        
        # Depthwise Separable Convolution
        x = F.relu(self.conv2(x))  # Output: 256x8x8
        
        # Flatten
        x = torch.flatten(x, 1)
        
        # Fully connected layer + Dropout
        x = F.relu(self.fc1(x))
        x = self.dropout(x)
        
        # Output layer
        x = self.fc2(x)
        
        return x

def get_model_from_name(model_name, number_of_classes=101):
    if model_name == "small":
        model = SmallConvNet(num_classes=number_of_classes)
    elif model_name == "simple":
        model = SimpleCNN(num_classes=number_of_classes)
    elif model_name == "improved":
        model = ImprovedCNN(num_classes=number_of_classes)
    elif model_name == "inception_v3":
        model = Inception_v3(num_classes=number_of_classes)
    elif model_name == "inception_v3_ft":
        model = Inception_v3_FT(num_classes=number_of_classes)
    else:
        raise ValueError("Invalid model name. Choose from 'small', 'simple', 'squeezenet'")
    return model

if __name__ == "__main__":
    parser = get_argparser()
    args = parser.parse_args()
    
    number_of_classes = args.number_of_classes
    image_input_share = args.image_input_share

    my_model = get_model_from_name(args.model_name)

    summary(my_model, image_input_share, device="cpu")