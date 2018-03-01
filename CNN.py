import torch
import torch.nn as nn
from torch.autograd import Variable
import torch.utils.data as Data
import torchvision      # 数据库模块

torch.manual_seed(1)    # reproducible

# Hyper Parameters
EPOCH = 1           # 训练整批数据多少次, 为了节约时间, 我们只训练一次
BATCH_SIZE = 50
LR = 0.001          # 学习率
DOWNLOAD_MNIST = False  # 如果你已经下载好了mnist数据就写上 Fasle


# Mnist 手写数字
train_data = torchvision.datasets.MNIST(
    root='./mnist/',    # 保存或者提取位置
    train=True,  # this is training data
    transform=torchvision.transforms.ToTensor(),    # 转换 PIL.Image or numpy.ndarray 成
                                                    # torch.FloatTensor (C x H x W), 训练的时候 normalize 成 [0.0, 1.0] 区间
    download=DOWNLOAD_MNIST,
)

test_data = torchvision.datasets.MNIST(
    root='./mnist/',
    train=False,
    download=DOWNLOAD_MNIST
)
train_loader = Data.DataLoader(
    dataset=train_data,
    batch_size=BATCH_SIZE,
    shuffle=True,
    num_workers=2
)


test_x = Variable(torch.unsqueeze(test_data.test_data, dim=1)).type(torch.FloatTensor)[:2000]

test_y = Variable(test_data.test_labels)[:2000]


class CNN(nn.Module):
    def __init__(self):
        super(CNN, self).__init__()
        self.conv1 = nn.Sequential(  # input shape (1, 28, 28)
            nn.Conv2d(
                in_channels=1,      # input height  输入图片的通道数 灰图只有一个通道, RGB有四个
                out_channels=16,    # n_filters     卷积核的个数
                kernel_size=5,      # filter size   卷积核尺寸 5*5
                stride=1,           # filter movement/step 卷积核移动步长
                padding=2,          # 外加padding大小
            ),      # output shape (16, 28, 28)
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2),    # 在 2x2 空间里向下采样, output shape (16, 14, 14), 默认 stride = kernel_size
        )
        self.conv2 = nn.Sequential(  # input shape (1, 28, 28)
            nn.Conv2d(16, 32, 5, 1, 2),  # output shape (32, 14, 14)
            nn.ReLU(),  # activation
            nn.MaxPool2d(2),  # output shape (32, 7, 7)
        )
        self.out = nn.Linear(32 * 7 * 7, 10)   # fully connected layer, output 10 classes

    def forward(self, x):
        x = self.conv1(x)
        x = self.conv2(x)
        x = x.view(x.size(0), -1)   # 展平多维的卷积图成 (32 * 7 * 7)
        # 此时的输出x是包含batchsize维度为4的tensor，即(batchsize,channels,x,y), 
        # x.size(0) 指batchsize的值.
        # x = x.view(x.size(0), -1)可简化为x = x.view(batchsize, -1)
        # view()函数的功能与reshape类似,用来转换size大小
        # x = x.view(batchsize, -1)中batchsize指转换后有几行, 而-1指在不告诉函数有多少列的情况下，根据原tensor数据和batchsize自动分配列数。
        output = self.out(x)
        return output


cnn = CNN()
optimizer = torch.optim.Adam(cnn.parameters(), lr=LR)
loss_func = torch.nn.CrossEntropyLoss()

for epoch in range(EPOCH):
    for step, (x, y) in enumerate(train_loader):
        b_x = Variable(x)
        b_y = Variable(y)

        output = cnn(b_x)
        loss = loss_func(output, b_y)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()


test_output = cnn(test_x[:10])
pred_y = torch.max(test_output, 1)[1].data.numpy().squeeze()
print(pred_y, 'prediction number')
print(test_y[:10].data.numpy(), 'real number')
