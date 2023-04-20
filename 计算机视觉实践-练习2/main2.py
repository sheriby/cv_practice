import torch
import torchvision
from torch import nn
from torch import optim
from lenet5 import LeNet

transform = torchvision.transforms.ToTensor()  
# transform = transforms.Compose([
#     transforms.ToTensor(),
#     transforms.Normalize(mean = (0.1307,), std = (0.3081,))
# ])

train_data = torchvision.datasets.MNIST(root="./data",    
                                        train=True,                                     
                                        transform=transform,                          
                                        download=True)                                 
test_data = torchvision.datasets.MNIST(root="./data",
                                        train=False,
                                        transform=transform,
                                        download=True)
train_loader = torch.utils.data.DataLoader(dataset = train_data, batch_size = 64,shuffle = True) 
test_loader = torch.utils.data.DataLoader(dataset = test_data, batch_size = 64,shuffle = False)

#define loss
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")   #若检测到GPU环境则使用GPU，否则使用CPU
net = LeNet().to(device)    #实例化网络，有GPU则将网络放入GPU加速
loss_fuc = nn.CrossEntropyLoss()    #多分类问题，选择交叉熵损失函数
optimizer = optim.SGD(net.parameters(),lr = 0.001,momentum = 0.9)   #选择SGD，学习率取0.001

#开始训练
EPOCH = 8   #迭代次数
for epoch in range(EPOCH):
    sum_loss = 0
    #数据读取
    for i,data in enumerate(train_loader):
        inputs,labels = data
        inputs, labels = inputs.to(device), labels.to(device)   #有GPU则将数据置入GPU加速
 
        # 梯度清零
        optimizer.zero_grad()
 
        # 传递损失 + 更新参数
        output = net(inputs)
        loss = loss_fuc(output,labels)
        loss.backward()
        optimizer.step()
 
        # 每训练100个batch打印一次平均loss
        sum_loss += loss.item()
        if i % 100 == 99:
            print('[Epoch:%d, batch:%d] train loss: %.03f' % (epoch + 1, i + 1, sum_loss / 100))
            sum_loss = 0.0
 
    correct = 0
    total = 0
 
    for data in test_loader:
        test_inputs, labels = data
        test_inputs, labels = test_inputs.to(device), labels.to(device)
        outputs_test = net(test_inputs)
        _, predicted = torch.max(outputs_test.data, 1)  #输出得分最高的类
        total += labels.size(0) #统计50个batch 图片的总个数
        correct += (predicted == labels).sum()  #统计50个batch 正确分类的个数
 
    print('第{}个epoch的识别准确率为：{}%'.format (epoch + 1, 100*correct.item()/total))

#模型保存
torch.save(net.state_dict(),'ckpt.mdl') 

#模型加载
#net.load_state_dict(torch.load('ckpt.mdl'))
