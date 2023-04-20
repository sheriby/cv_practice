import torch
import torch.nn as nn

class LeNet(nn.Module):
    def __init__(self):
        super(LeNet, self).__init__()
        #输入:(6*28*28);输出：(6*14*14)
        self.conv1 = nn.Sequential(     
            nn.Conv2d(1,6,kernel_size=5,stride=1,padding=2), #新建卷积层,kernel_size表示卷积核大小,stride表示步长,padding=2，图片大小变为 28+2*2 = 32 (两边各加2列0)，保证输入输出尺寸相同
            nn.ReLU(),
            nn.AvgPool2d(kernel_size=2 ,stride=2,padding=0)   #input_size=(6*28*28)，output_size=(6*14*14)
        )

        #输入:(6*14*14);输出：(16*5*5)
        self.conv2 = nn.Sequential(
            nn.Conv2d(6,16,kernel_size=5,stride=1,padding=0), #input_size=(6*14*14)，output_size=16*10*10
            nn.ReLU(),
            nn.AvgPool2d(kernel_size=2,stride=2,padding=0)    ##input_size=(16*10*10)，output_size=(16*5*5)
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
    
        x = x.view(x.size(0), -1) #全连接层均使用的nn.Linear()线性结构，输入输出维度均为一维，故需要把数据拉为一维
        x = self.fc1(x)
        x = self.fc2(x)
        x = self.fc3(x)
        return x