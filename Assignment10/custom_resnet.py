from __future__ import print_function
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torchvision import datasets, transforms
import albumentations as A
from albumentations.pytorch import ToTensorV2
from torchsummary import summary


import torch.nn.functional as F
dropout_value = 0.1
class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()

        self.convblock1 = nn.Sequential(
            nn.Conv2d(in_channels=3, out_channels=64, kernel_size=(3, 3), padding=1, stride =1, bias=False),
            nn.BatchNorm2d(64),
            nn.ReLU()
        )  #Prep Layer
     
    #Layer 1 
        self.convblock2 = nn.Sequential(
            nn.Conv2d(in_channels=64, out_channels=128, kernel_size=(3, 3), padding=1, bias=False),
            nn.MaxPool2d(kernel_size=2, stride=2), #Feature map size became 16
            nn.BatchNorm2d(128),
            nn.ReLU(),

        )  

        self.resblock1 = nn.Sequential(
            nn.Conv2d(in_channels=128, out_channels=128, kernel_size=(3, 3), padding=1, bias=False),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.Conv2d(in_channels=128, out_channels=128, kernel_size=(3, 3), padding=1, bias=False), 
            nn.BatchNorm2d(128),
            nn.ReLU(),           
        )

    #Layer 2
        self.convblock3 = nn.Sequential(
            nn.Conv2d(in_channels=128, out_channels=256, kernel_size=(1, 1), padding=1, bias=False),
            nn.MaxPool2d(kernel_size=2, stride=2), #Feature map size became 8
            nn.BatchNorm2d(256),
            nn.ReLU(),
        ) 

#Layer 3

        self.convblock4 = nn.Sequential(
            nn.Conv2d(in_channels=256, out_channels=512, kernel_size=(3, 3), padding=1,stride=1, bias=False),
            nn.ReLU(),            
            nn.BatchNorm2d(512),
        ) 
        self.resblock2 = nn.Sequential(
            nn.Conv2d(in_channels=512, out_channels=512, kernel_size=(3, 3), padding=1, stride=1, bias=False),
            nn.BatchNorm2d(512),
            nn.ReLU(),
            nn.Conv2d(in_channels=512, out_channels=512, kernel_size=(3, 3), padding=1,stride=1, bias=False),  
            nn.BatchNorm2d(512),
            nn.ReLU(),          
        )
        self.pool2 = nn.MaxPool2d(4, 2) # 3x3, FC will layer will get 512x3x3

        self.fc1 = nn.Linear(4608, 10, bias=False)
        # self.fc2 = nn.Linear(2000, 10, bias=False)
     

    def forward(self, x):
        x = self.convblock1(x)
        x = self.convblock2(x)
        x = x + self.resblock1(x)

        x = self.convblock3(x)

        x = self.convblock4(x)
        x = x + self.resblock2(x)
        x = self.pool2(x)
        x = x.view(-1, 4608)
        x = self.fc1(x)
        # x= self.fc2(x)

        return F.log_softmax(x, dim=-1)
    
