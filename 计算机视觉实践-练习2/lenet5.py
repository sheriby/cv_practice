import torch
import torch.nn as nn

class LeNet(nn.Module):
    def __init__(self):
        super(LeNet, self).__init__()
        #输入:(6*28*28);输出：(6*14*14)
        self.conv1 = nn.Sequential(     
            nn.Conv2d(1,6,kernel_size=5,stride=1,padding=2), 
            nn.BatchNorm2d(6),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size = 2 ,stride = 2,padding=0)
        )

        self.conv2 = nn.Sequential(
            nn.Conv2d(6,6,kernel_size=3,stride=1,padding=1), 
            nn.BatchNorm2d(6),
            nn.ReLU(),
        )

        self.conv3 = nn.Sequential(
            nn.Conv2d(6,6,kernel_size=3,stride=1,padding=1), 
            nn.BatchNorm2d(6),
            nn.ReLU(),
        )

        self.conv4 = nn.Sequential(
            nn.Conv2d(6,16,kernel_size=5,stride=1,padding=0),
            nn.BatchNorm2d(16),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2,stride = 2,padding=0) 
        )
        #全连接层
        self.fc1 = nn.Sequential(
            nn.Linear(16*5*5,120),
            nn.ReLU()
        )
        #全连接层
        self.fc2 = nn.Sequential(
            nn.Linear(120,84),
            nn.ReLU()
        )
        #全连接层
        self.fc3 = nn.Linear(84,10)

    #网络前向传播过程
    def forward(self, x):
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.conv3(x)
        x = self.conv4(x)
    
        x = x.view(x.size(0), -1) 
        x = self.fc1(x)
        x = self.fc2(x)
        x = self.fc3(x)
        return x