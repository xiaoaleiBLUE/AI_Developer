import torch
from torch import nn
import numpy as np
import pandas as pd
import datetime
import matplotlib.pyplot as plt
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torch.utils import data
import torchvision
from torchvision.transforms import ToTensor, transforms
import glob
import os, PIL, random, pathlib
from PIL import Image
from lxml import etree
from matplotlib.patches import Rectangle                                # 绘制矩形框
from torch.optim import lr_scheduler

# 分类: 模型预测图片中有什么对象, 最基础
# 分类+定位: 图像的对象是什么, 确定该对象所处的位置
# 语义分割: 区分到图中的每一像素点, 而不仅仅是矩形框框住
# 目标检测: 把他们用矩形框框住
# 实例分割: 目标检测和语义分割的结合

# 图像定位: 输出四个数字(x,y,w, h), 图像中某一点的坐标(x, y), 以及图像的宽度和高度
pil_img = Image.open(r'./dataset3/images/Abyssinian_1.jpg')
np_img = np.array(pil_img)                                              # 转换成numpy才能绘图, 也可直接打开图片绘图
print(np_img.shape)                                                     # (400, 600, 3)
plt.imshow(pil_img)
plt.show()

xml_img = open(r'./dataset3/annotations/xmls/Abyssinian_1.xml').read()
sel = etree.HTML(xml_img)                                               # 解析文件

width = sel.xpath('//size/width/text()')[0]                             # //: 根目录, text():文本
print(width)                                                            # 600
width = int(width)

height = sel.xpath('//size/height/text()')[0]
print(height)                                                           # 400
height = int(height)

xmin = sel.xpath('//object/bndbox/xmin/text()')[0]
print(xmin)                                                             # 333

xmax = sel.xpath('//object/bndbox/xmax/text()')[0]
print(xmax)                                                             # 425

ymin = sel.xpath('//object/bndbox/ymin/text()')[0]
print(ymin)                                                             # 72

ymax = sel.xpath('//object/bndbox/ymax/text()')[0]
print(ymax)                                                             # 158

xmin = int(xmin)
xmax = int(xmax)
ymin = int(ymin)
ymax = int(ymax)

# 绘制矩形框, Rectangle参数: xy, width, height, fill=False:矩形框不填充
plt.imshow(np_img)
rect = Rectangle((xmin, ymin), (xmax - xmin), (ymax - ymin), fill=False, color='red')
ax = plt.gca()                                                          # 获取当前坐标系
ax.axes.add_patch(rect)                                                 # 在当前坐标系添加矩形框
plt.show()

# 改变图片大小, 比例进行缩放坐标, 获取当前坐标系, 并在坐标系中添加矩形框
img = pil_img.resize((224, 224))
xmin = (xmin/width)*224
ymin = (ymin/height)*224
xmax = (xmax/width)*224
ymax = (ymax/height)*224

plt.imshow(img)
rect = Rectangle((xmin, ymin), (xmax - xmin), (ymax - ymin), fill=False, color='purple')
ax = plt.gca()
ax.axes.add_patch(rect)
plt.show()


# 创建输入
all_img_path = glob.glob(r'./dataset3/images/*.jpg')
print(all_img_path)
print(len(all_img_path))                                                # 7390

all_img_anno = glob.glob(r'./dataset3/annotations/xmls/*.xml')
print(all_img_anno)
print(len(all_img_anno))                                                # 3686

all_anno_name = [name.split('\\')[1].replace('.xml', '') for name in all_img_anno]
print(all_anno_name)
# 下面这种方法也可以实现上述功能
# all_anno_name = [name.split('\\')[1].split('.')[0] for name in all_img_anno]
# print(all_anno_name)

# 如果图片路径中的图片名称在all_anno_name中, 我们就要这张图片的路径, 此时imgs为标签对应的图片路径
# imgs 为列表形式
imgs = [imgs for imgs in all_img_path
         if imgs.split('\\')[1].replace('.jpg', '') in all_anno_name]
print(imgs)
print(len(imgs))                                                        # 3686


# 将 xml 中的数值解析出来
# 希望输入路径进去, 解析出坐标比例出来
# 防止转义, 可用format()方法,  xml = open(r'{}'.format(path)).read()
def path_to_labels(path):
    xml = open(path).read()                                            # 打开xml文件
    sel = etree.HTML(xml)
    width = int(sel.xpath('//size/width/text()')[0])
    height = int(sel.xpath('//size/height/text()')[0])
    xmin = int(sel.xpath('//object/bndbox/xmin/text()')[0])
    xmax = int(sel.xpath('//object/bndbox/xmax/text()')[0])
    ymin = int(sel.xpath('//object/bndbox/ymin/text()')[0])
    ymax = int(sel.xpath('//object/bndbox/ymax/text()')[0])

    return [xmin/width, ymin/height, xmax/width, ymax/height]          # 返回四个坐标比例值


labels = [path_to_labels(label_xml) for label_xml in all_img_anno]
print(labels)                                                          # labels列表对应标签的坐标比例值

# 进行乱序索引
index = np.random.permutation(len(imgs))
imgs = np.array(imgs)[index]
labels = np.array(labels)[index]
print(labels.shape)                                          # (3686, 4)
lables = labels.astype(np.float32)
print(lables[4])                                             # [0.082  0.00301205  0.534  0.7078313 ]
# 划分数据集: 训练, 测试
i = int(len(imgs)*0.8)
train_imgs = imgs[:i]
train_labels = labels[:i]
test_imgs = imgs[i:]
test_labels = labels[i:]

transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
])


class Data_oxford(data.Dataset):
    def __init__(self, imgs_path, labels_list):
        super(Data_oxford, self).__init__()
        self.imgs = imgs_path
        self.labels = labels_list
        self.transform = transform

    def __getitem__(self, index):
        img = self.imgs[index]                         # 切出来是图片的路径, 可使用Image.open()进行打开
        # label = self.labels[index]                     # 对labels列表进行切片, 切出来是坐标值
        pil_img = Image.open(img)
        pil_img = pil_img.convert('RGB')
        img_tensor = transform(pil_img)
        l1, l2, l3, l4 = self.labels[index]
        return img_tensor, l1, l2, l3, l4

    def __len__(self):
        return len(self.imgs)


train_dataset = Data_oxford(train_imgs, train_labels)
test_dataset = Data_oxford(test_imgs, test_labels)

train_dataloader = DataLoader(train_dataset, batch_size=32, shuffle=True)
test_dataloader = DataLoader(test_dataset, batch_size=32, shuffle=False)

img_batch, xmin_batch, ymin_batch, xmax_batch, ymax_batch = next(iter(train_dataloader))
plt.figure(figsize=(12, 8))
for i, (img, xmin, ymin, xmax, ymax) in enumerate(zip(img_batch[:6], xmin_batch[:6],
                                                      ymin_batch[:6], xmax_batch[:6], ymax_batch[:6])):
    img = img.permute(1, 2, 0).numpy()
    plt.subplot(2, 3, i+1)
    plt.imshow(img)
    xmin = xmin*224
    ymin = ymin*224
    xmax = xmax*224
    ymax = ymax*224
    rect = Rectangle((xmin, ymin), (xmax - xmin), (ymax - ymin), fill=False, color='red')
    ax = plt.gca()
    ax.axes.add_patch(rect)
plt.show()


#  =============
# 创建定位模型, 卷积基提取特征模型
resnet = torchvision.models.resnet101(pretrained=True)             # 使用resnet初始化参数
print(resnet)
# list()返回所有层的生成器,
print(len(list(resnet.children())))                                # 10

# list()是列表形式, 可对所有层的生成器进行切片查看每一层的相关信息
# 查看最后一层的信息: Linear(in_features=2048, out_features=1000, bias=True)
print(list(resnet.children())[-1])                                 # 输出最后一层

# 取出除全连接以外的所有层
list(resnet.children())[:-1]

# 除全连接层外的所有层封装在一起
# 1.创建一个层, 将取出除全连接以外的所有层存放在里面
# 2.对list进行解包, *: 解包
print('======='*8)
conv_base = nn.Sequential(*list(resnet.children())[:-1])
print(conv_base)
in_size = resnet.fc.in_features


# 四个全连接层, 用来输出四个坐标值
class Ox_net(nn.Module):
    def __init__(self):
        super(Ox_net, self).__init__()
        self.conv = conv_base            # self.conv = nn.Sequential(*list(resnet.children())[:-1])也行
        self.fc_1 = nn.Linear(in_size, 1)
        self.fc_2 = nn.Linear(in_size, 1)
        self.fc_3 = nn.Linear(in_size, 1)
        self.fc_4 = nn.Linear(in_size, 1)

    def forward(self, x):
        x = self.conv(x)
        x1 = self.fc_1(x)
        x2 = self.fc_2(x)
        x3 = self.fc_3(x)
        x4 = self.fc_4(x)
        return x1, x2, x3, x4


model = Ox_net().to('cuda')

# 图像定位问题, 本质是一个回归问题, 采用回归损失函数
# 回归问题一般没有 acc
loss_fn = nn.MSELoss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.0001)
exp_lr_scheduler = lr_scheduler.StepLR(optimizer, step_size=10, gamma=0.1)


# 放进训练行, 与之前的训练函数略有不同
def fit(epoch, model, train_dataloader, test_dataloader):
    total = 0.0
    running_loss = 0.0
    model.train()
    for x, y1, y2, y3, y4 in train_dataloader:
        if torch.cuda.is_available():
            x, y1, y2, y3, y4 = (x.to('cuda'), y1.to('cuda'), y2.to('cuda'),
                                 y3.to('cuda'), y4.to('cuda'))

        y_pre1, y_pre2, y_pre3, y_pre4 = model(x)
        loss_1 = loss_fn(y_pre1, y1)
        loss_2 = loss_fn(y_pre2, y2)
        loss_3 = loss_fn(y_pre3, y3)
        loss_4 = loss_fn(y_pre4, y4)
        loss = loss_1 + loss_2 + loss_3 + loss_4
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        with torch.no_grad():
            running_loss += loss.item()
    exp_lr_scheduler.step()
    train_epoch_loss = running_loss / len(train_dataloader.dataset)


    test_total = 0.0
    test_running_loss = 0.0
    model.eval()
    with torch.no_grad():
        for x, y1, y2, y3, y4 in train_dataloader:
            if torch.cuda.is_available():
                x, y1, y2, y3, y4 = (x.to('cuda'), y1.to('cuda'), y2.to('cuda'),
                                     y3.to('cuda'), y4.to('cuda'))

            y_pre1, y_pre2, y_pre3, y_pre4 = model(x)
            loss_1 = loss_fn(y_pre1, y1)
            loss_2 = loss_fn(y_pre2, y2)
            loss_3 = loss_fn(y_pre3, y3)
            loss_4 = loss_fn(y_pre4, y4)
            loss = loss_1 + loss_2 + loss_3 + loss_4
            test_running_loss += loss.item()

    epoch_test_loss = test_running_loss / len(test_dataloader.dataset)
    print('epoch: ', epoch,
          'loss: ', round(train_epoch_loss, 3),
          'test_loss: ', round(epoch_test_loss, 3),
          )

    return train_epoch_loss, epoch_test_loss


epochs = 20
train_loss = []
test_loss = []

for epoch in range(epochs):
    epoch_train_loss, epoch_test_loss = fit(epoch, model, train_dataloader, test_dataloader)
    train_loss.append(epoch_train_loss)
    test_loss.append(epoch_test_loss)


# 模型保存
path = './path'
if os.path.isdir(path):
    os.makedirs(path)

torch.save(model.state_dict(), path)

# 进行预测
plt.figure(figsize=(12, 8))
imgs, _, _, _, _ = next(iter(test_dataloader))                    # _, _, _, _占位符, 表示数值不要了
imgs = imgs.to('cuda')
out1, out2, out3, out4 = model(imgs)
for i in range(6):
    plt.subplot(2, 3, i+1)
    plt.imshow(imgs[i].permute(1, 2, 0).cpu().numpy())
    xmin, ymin, xmax, ymax = (out1[i].item()*224,
                              out2[i].item()*224,
                              out3[i].item()*224,
                              out4[i].item()*224)
    rect = Rectangle((xmin, ymin), (xmax - xmin), (ymax - ymin), fill=False, color='red')
    ax = plt.gca()
    ax.axes.add_patch(rect)
plt.show()



















