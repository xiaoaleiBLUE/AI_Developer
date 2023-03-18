"""
P5: 运动鞋识别
还需要改网络结构, 调整参数
"""
import torch
from torch import nn
import numpy as np
import pandas as pd
import datetime
import matplotlib.pyplot as plt
import torch.nn.functional as F
from torch.utils.data import DataLoader
import torchvision
from torchvision.transforms import ToTensor, transforms
import glob
import os, PIL, random, pathlib
from PIL import Image

transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

train_dataset = torchvision.datasets.ImageFolder(root=r'./46-data/train',
                                                transform=transform)
test_dataset = torchvision.datasets.ImageFolder(root=r'./46-data/test',
                                                transform=transform)

print(train_dataset.class_to_idx)
idx_to_class = dict((v, k) for k, v in train_dataset.class_to_idx.items())
print(idx_to_class)


train_dataloader = DataLoader(train_dataset, batch_size=32, shuffle=True)
test_dataloader = DataLoader(test_dataset, batch_size=32, shuffle=True)


for x, y in test_dataloader:
    print('x,shape:', x.shape)
    print('y.shape', y.shape)
    break

imgs, labels = next(iter(train_dataloader))

plt.figure(figsize=(12, 8))                                # 画布大小
for i, (img, label) in enumerate(zip(imgs[:6], labels[:6])):
    img = img.permute(1, 2, 0).numpy()
    img = (img + 1) / 2
    plt.subplot(2, 3, i+1)
    plt.title(idx_to_class[label.item()])
    plt.imshow(img)
plt.show()


def printlog(info):
    nowtime = datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')
    print("\n"+"========="*8 + '%s'%nowtime)
    print(str(info)+"\n")


class Net_work(nn.Module):
    def __init__(self):
        super(Net_work, self).__init__()
        self.conv_1 = nn.Sequential(
            nn.Conv2d(in_channels=3, out_channels=12, kernel_size=5, padding=0),       # 12*220*220
            nn.BatchNorm2d(12),
            nn.ReLU(inplace=True)
        )
        self.conv_2 = nn.Sequential(
            nn.Conv2d(in_channels=12, out_channels=12, kernel_size=5, padding=0),      # 12*216*216
            nn.BatchNorm2d(12),
            nn.ReLU(inplace=True)
        )
        self.poo_1 = nn.MaxPool2d(2, 2)                                                # 12*108*108
        self.conv_3 = nn.Sequential(
            nn.Conv2d(in_channels=12, out_channels=24, kernel_size=5, padding=0),      # 24*104*104
            nn.BatchNorm2d(24),
            nn.ReLU(inplace=True)
        )
        self.conv_4 = nn.Sequential(
            nn.Conv2d(in_channels=24, out_channels=24, kernel_size=5, padding=0),      # 24*100*100
            nn.BatchNorm2d(24),
            nn.ReLU(inplace=True)
        )
        self.pool_2 = nn.MaxPool2d(2, 2)                                               # 24*50*50
        self.conv_5 = nn.Sequential(
            nn.Conv2d(in_channels=24, out_channels=48, kernel_size=5, padding=0),      # 48*46*46
            nn.BatchNorm2d(48),
            nn.ReLU()
        )
        self.conv_6 = nn.Sequential(
            nn.Conv2d(in_channels=48, out_channels=48, kernel_size=5, padding=0),      # 48*42*42
            nn.BatchNorm2d(48),
            nn.ReLU()
        )
        self.pool_3 = nn.MaxPool2d(2, 2)                                               # 48*21*21
        self.linear_1 = nn.Linear(48*21*21, 2)

    def forward(self, x):
        x = self.conv_1(x)
        x = self.conv_2(x)
        x = self.poo_1(x)
        x = self.conv_3(x)
        x = self.conv_4(x)
        x = self.pool_2(x)
        x = self.conv_5(x)
        x = self.conv_6(x)
        x = self.pool_3(x)
        x = x.view(-1, 48*21*21)
        x = self.linear_1(x)
        return x


device = 'cuda' if torch.cuda.is_available() else 'cpu'
print(device)
model = Net_work().to(device)


def adjust_learing_rate(optimizer, epoch, learn_rate):
    # 每 2 个epoch衰减到原来的 0.98
    lr = learn_rate*(0.98**(epoch // 2))
    for p in optimizer.param_groups:
        p['lr'] = lr


# 与上述方法等价: 调用官方动态学习率接口时使用
# lambda1 = lambda epoch: (0.98 ** (epoch // 2))
# optimizer = torch.optim.SGD(model.parameters(), lr=0.0001)
# scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=lambda1)     # 选用的调整方法

optimizer = torch.optim.Adam(model.parameters(), lr=0.0001)    # lr: 初始学习率
loss_fn = nn.CrossEntropyLoss()


def train(train_dataloader, model, loss_fn, optimizer):
    size = len(train_dataloader.dataset)                     # 训练集的大小
    num_of_batch = len(train_dataloader)                     # 返回批次数目: size/batch_size 向上取整
    train_correct, train_loss = 0.0, 0.0
    for x, y in train_dataloader:
        x, y = x.to(device), y.to(device)
        pre = model(x)
        loss = loss_fn(pre, y)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        with torch.no_grad():
            train_correct += (pre.argmax(1) == y).type(torch.float).sum().item()
            train_loss += loss.item()

    train_correct /= size
    train_loss /= num_of_batch
    return train_correct, train_loss


def test(test_dataloader, model, loss_fn):
    size = len(test_dataloader.dataset)                       # 训练集的大小
    num_of_batch = len(test_dataloader)                       # 返回批次数目: size/batch_size 向上取整
    test_correct, test_loss = 0.0, 0.0
    with torch.no_grad():
        for x, y in test_dataloader:
            x, y = x.to(device), y.to(device)
            pre = model(x)
            loss = loss_fn(pre, y)
            test_loss += loss.item()
            test_correct += (pre.argmax(1) == y).type(torch.float).sum().item()

    test_correct /= size
    test_loss /= num_of_batch
    return test_correct, test_loss


epochs = 100
train_loss = []
train_acc = []
test_loss = []
test_acc = []
for epoch in range(epochs):
    printlog('Epoch {0} / {1}'.format(epoch, epochs))
    adjust_learing_rate(optimizer, epoch, 0.0001)
    # scheduler.step()                                  # 更新学习率(调用官方动态学习率接口时使用)

    model.train()
    epoch_train_acc, epoch_train_loss = train(train_dataloader, model, loss_fn, optimizer)
    model.eval()
    epoch_test_acc, epoch_test_loss = test(test_dataloader, model, loss_fn)
    train_loss.append(epoch_train_loss)
    train_acc.append(epoch_train_acc)
    test_loss.append(epoch_test_loss)
    test_acc.append(epoch_test_acc)

    # 获取当前学习率
    lr = optimizer.state_dict()['param_groups'][0]['lr']

    template = ('train_loss: {:.5f}, train_acc: {:.5f}, test_loss: {:.5f}, test_acc: {:.5f}, Lr:{:.2E}')
    print(template.format(epoch_train_loss, epoch_train_acc, epoch_test_loss, epoch_test_acc, lr))

print('done')

plt.plot(range(epochs), train_loss, label='train_loss')
plt.plot(range(epochs), train_acc, label='train_acc')
plt.plot(range(epochs), test_loss, label='test_loss')
plt.plot(range(epochs), test_acc, label='test_acc')
plt.legend()
plt.show()
print('done')

# 指定图片进行预测
classes = list(train_dataset.class_to_idx)
print(classes)


def perdict_image(image_path, model, transforms, classes):
    test_img = Image.open(image_path).convert('RGB')
    test_img = transforms(test_img)
    img = test_img.to(device).unsqueeze(0)

    model.eval()
    output = model(img)

    _, pred = torch.max(output, 1)
    pre_class = classes[pred]
    print('预测结果是: ', pre_class)


perdict_image(image_path=r'./46-data/train/nike/1 (14).jpg',
              model=model,
              transforms=transform,
              classes=classes)


# 保存并加载模型
# path = './week_5.pth'
# torch.save(model.state_dict(), path)
#
#
# 将参数加载到模型中
# model.load_state_dict(torch.load(path, map_location=device))








