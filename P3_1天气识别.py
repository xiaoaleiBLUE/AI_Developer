# 四种天气图片的数据分类
# torchvision.dataset.ImageFolder: 从分类文件夹中创建dataset数据
import torch
from torch import nn
import datetime
from torch.utils.data import DataLoader
import torch.nn.functional as F
import torchvision
from torchvision.transforms import transforms
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import os
import shutil

base_dir = './4_weather'
if not os.path.isdir(base_dir):
    os.makedirs(base_dir)                                 # 如果没有base_dir, 就创建base_dir
    train_dir = os.path.join(base_dir, 'train')           # 在base_dir下加入train目录
    test_dir = os.path.join(base_dir, 'test')
    os.makedirs(train_dir)                                # 创建 train_dir 目录
    os.makedirs(test_dir)

# train文件夹下创建cloudy, rain, shine, sunrise文件夹
specises = ['cloudy', 'rain', 'shine', 'sunrise']
cloudt_dir = './4_weather/train/cloudy'
if not os.path.isdir(cloudt_dir):
    for train_or_test in ['train', 'test']:
        for spec in specises:
            os.makedirs(os.path.join(base_dir, train_or_test, spec))

# os.listdir(image_dir)                                    # 列出image_dir文件夹所有文件的名称
image_dir = './dataset2'
for i, img_name in enumerate(os.listdir(image_dir)):
    for spec in specises:
        if spec in img_name:                               # 如果 spec 在 img_name 中
                s = os.path.join(image_dir, img_name)
                if i%5==0:
                    d = os.path.join(base_dir, 'test', spec, img_name)
                else:
                    d = os.path.join(base_dir, 'train', spec, img_name)
                shutil.copy(s, d)                          # s拷贝到d

# 查看每个cloudy, rain, shine, sunrise文件下图片的个数
for train_or_test in ['train', 'test']:
    for spec in specises:
        print(train_or_test, spec, len(os.listdir(os.path.join(base_dir, train_or_test, spec))))

transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
])

# torchvision.dataset.ImageFolder: 从分类文件夹中创建dataset数据, 文件为train, test
# train_dataset的root 为 train 目录, 即 4_weather下的train目录, train目录下为类别目录
train_dataset = torchvision.datasets.ImageFolder(root='./4_weather/train',
                                                 transform=transform
                                                 )

test_dataset = torchvision.datasets.ImageFolder(root='./4_weather/test',
                                                transform=transform
                                                )

# train_dataset.classes: 根据分的文件夹的名字来确定的类别
# train_dataset.class_to_idx: 按顺序为这些类别定义索引为0, 1...
print(train_dataset.classes)                               # ['cloudy', 'rain', 'shine', 'sunrise']
print(train_dataset.class_to_idx)                          # {'cloudy': 0, 'rain': 1, 'shine': 2, 'sunrise': 3}

train_dataloader = DataLoader(train_dataset, batch_size=32, shuffle=True)
test_dataloader = DataLoader(test_dataset, batch_size=32, shuffle=False)

imgs, labels = next(iter(train_dataloader))
print(imgs.shape)                                          # torch.Size([32, 3, 224, 224])

id_to_class = dict((v, k) for k, v in train_dataset.class_to_idx.items())
print(id_to_class)                                         # {0: 'cloudy', 1: 'rain', 2: 'shine', 3: 'sunrise'}
print(id_to_class[1])

plt.figure(figsize=(12, 8))
for i, (img, label) in enumerate(zip(imgs[:6], labels[:6])):
    img = img.permute(1, 2, 0).numpy()
    img = (img + 1) / 2
    plt.subplot(2, 3, i+1)
    plt.title(id_to_class[label.item()])                   # plt.title(id_to_class.get(label.item()))
    plt.imshow(img)
plt.show()


def printlog(info):
    nowtime = datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')
    print("\n"+"========="*8 + '%s'%nowtime)
    print(str(info)+"\n")



class Net(nn.Module):
    def __int__(self):
        super(Net, self).__init__()
        self.conv_1 = nn.Conv2d(3, 16, 3)
        self.conv_2 = nn.Conv2d(16, 32, 3)
        self.conv_3 = nn.Conv2d(32, 64, 3)
        self.pool = nn.MaxPool2d(2, 2)
        self.fc_1 = nn.Linear(64*10*10, 1024)
        self.fc_2 = nn.Linear(1024, 4)

    def forward(self, x):
        x = F.relu(self.conv_1(x))
        x = self.pool(x)
        x = F.relu(self.conv_2(x))
        x = self.pool(x)
        x = F.relu(self.conv_3(x))
        x = self.pool(x)
        x = x.view(-1, 64*10*10)
        x = F.relu(self.fc_1(x))
        x = self.fc_2(x)
        return x


device = 'cuda' if torch.cuda.is_available() else 'cpu'
model = Net().to(device)
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
loss_fn = nn.CrossEntropyLoss()


def train(train_dataloader, model, loss_fn, optimizer):
    size = len(train_dataloader.dataset)
    num_of_batch = len(train_dataloader)
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
    size = len(test_dataloader.dataset)
    num_of_batch = len(test_dataloader)
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
    model.train()
    if epoch % 10 == 0:
        for p in optimizer.param_groups:
            p['lr'] *= 0.9
    epoch_train_acc, epoch_train_loss = train(train_dataloader, model, loss_fn, optimizer)
    model.eval()
    epoch_test_acc, epoch_test_loss = test(test_dataloader, model, optimizer)
    train_loss.append(epoch_train_loss)
    train_acc.append(epoch_train_acc)
    test_loss.append(epoch_test_loss)
    test_acc.append(epoch_test_acc)

    template = ('train_loss: {:.5f}, train_acc: {:.5f}, test_loss: {:.5f}, test_acc: {:.5f}')
    print(template.format(epoch_train_loss,epoch_train_acc, epoch_test_loss, epoch_test_acc))
print('done')

plt.plot(range(epochs), train_loss, label='train_loss')
plt.plot(range(epochs), train_acc, label='train_acc')
plt.plot(range(epochs), test_loss, label='test_loss')
plt.plot(range(epochs), test_acc, label='test_acc')
print('done')



























