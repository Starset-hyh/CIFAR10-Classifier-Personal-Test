import torchvision  #准备阶段：pip install pytorch
from torch.utils.data import DataLoader
import torch
from torch import nn
from torch.utils.tensorboard import SummaryWriter
from torch import argmax

train_data = torchvision.datasets.CIFAR10(root="../train_data",train=True, transform=torchvision.transforms.ToTensor(),download=True)
test_data = torchvision.datasets.CIFAR10(root='../test_data',train=False,transform=torchvision.transforms.ToTensor(),download=True)

train_data_size = len(train_data)
test_data_size = len(test_data)

print('训练数据集的长度为：{}'.format(train_data_size)) #查看数据集大小
print('测试数据集的长度为：{}'.format(test_data_size))
print('CUDA是否可用: {torch.cuda.is_available()}')
print("PyTorch版本: {torch.__version__}")
if torch.cuda.is_available():
    print(f"CUDA版本: {torch.version.cuda}")
    print(f"GPU设备名称: {torch.cuda.get_device_name(0)}")
else:
    print("❌ GPU不可用")

train_dataloader = DataLoader(train_data, batch_size=100)
test_dataloader = DataLoader(test_data, batch_size=100)

class cifar10(nn.Module): #神经网络
    def __init__(self):
        super(cifar10, self).__init__()
        self.model = nn.Sequential(
            nn.Conv2d(3, 32, 5, 1, 2), #二维图像卷积层，函数内结构（输入通道数，输出通道数，卷积核尺寸，步长，padding，空洞卷积的元素间距，分组连接，是否加入偏置，padding的模式）
            nn.MaxPool2d(2),           #卷积层结构只有前三个是必填，别的是选填
            nn.Conv2d(32, 32, 5, 1, 2),
            nn.MaxPool2d(2), #池化层，函数内结构（池化核尺寸）
            nn.Conv2d(32, 64, 5, 1, 2),
            nn.MaxPool2d(2),
            nn.Flatten(),
            nn.Linear(1024,64), #线性层，函数内结构（输入维数，输出维数） 
            nn.ReLU(), #ReLU激活函数
            nn.Linear(64,64),
            nn.ReLU(),
            nn.Linear(64,64),
            nn.ReLU(),
            nn.Linear(64,64),
            nn.Linear(64,10)
        )

    def forward(self, x):
        x = self.model(x)
        return(x)

if __name__ == '__main__':
    CIFAR10 = cifar10()
    input = torch.ones((100, 3, 32, 32))
    output = CIFAR10(input)
    print(output.shape)

CIFAR10 = cifar10()
loss_F = nn.CrossEntropyLoss() #损失函数（使用交叉熵函数 y = -Σylogyhat）
if torch.cuda.is_available():
    CIFAR10 = CIFAR10.cuda() #启用Nvidia显卡加速运算（RTX50系显卡似乎不可用）
    loss_F = loss_F.cuda()
optimizer = torch.optim.SGD(CIFAR10.parameters(), lr=1e-2) #优化器

total_training_step = 0 
total_testing_step = 0
epoch = 100
writer = SummaryWriter('../trainin_loss')

for i in range(epoch):
    i = i + 1
    print("<<<<<<<<< TRAINING ROUND {} >>>>>>>>>".format(i)) #轮次计数
    for data in train_dataloader:
        imgs, targets = data #标签结构：（图像，标签）
        if torch.cuda.is_available():
            imgs = imgs.cuda()
            targets = targets.cuda()
        outputs = CIFAR10(imgs)
        loss = loss_F(outputs, targets)

        optimizer.zero_grad()
        loss.backward() #梯度回传
        optimizer.step()

        total_training_step = total_training_step + 1
        if total_training_step % 100 == 0:
            print('训练次数：{}；损失：{}'.format(total_training_step, loss.item()))
            writer.add_scalar('train_loss', loss.item(), total_testing_step)

    total_testing_loss = 0
    total_acc = 0
    with torch.no_grad(): #测试步骤：取消梯度回传
        for data in test_dataloader:
            imgs, targets = data
            if torch.cuda.is_available():
                imgs = imgs.cuda()
                targets = targets.cuda()
            outputs = CIFAR10(imgs)
            loss = loss_F(outputs, targets)
            total_testing_loss = total_testing_loss + loss
            acc = (outputs.argmax(1) == targets).sum()
            total_acc = total_acc + acc
    print('测试集正确率：{}'.format(total_acc / test_data_size))
    print('测试集损失：{}'.format(total_testing_loss))
    writer.add_scalar('test_loss', total_testing_loss, total_testing_step)
    writer.add_scalar('test_accurancy', total_acc/test_data_size, total_testing_step)
    total_testing_step = total_testing_step + 1

