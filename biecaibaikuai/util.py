import os
import sys
import json
import pickle
import random

import torch
from tqdm import tqdm

import matplotlib.pyplot as plt
import torch.nn.functional as F
from torch.utils.tensorboard import SummaryWriter
writer = SummaryWriter('./mnist/tensorboard')

def read_split_data(root: str, val_rate: float = 0.2):
    random.seed(0)  # 保证随机结果可复现
    assert os.path.exists(root), "dataset root: {} does not exist.".format(root)
    if os.path.exists('./biecaibaikuai/json/train_images_path.json'):
        with open('./biecaibaikuai/json/train_images_path.json', 'r') as json_file:
            train_images_path = json.load(json_file)
        with open('./biecaibaikuai/json/train_images_label.json', 'r') as json_file:
            train_images_label = json.load(json_file)
        with open('./biecaibaikuai/json/val_images_path.json', 'r') as json_file:
            val_images_path = json.load(json_file)
        with open('./biecaibaikuai/json/val_images_label.json', 'r') as json_file:
            val_images_label = json.load(json_file)
        return train_images_path, train_images_label, val_images_path, val_images_label


    # 遍历文件夹，一个文件夹对应一个类别
    flower_class = [cla for cla in os.listdir(root) if os.path.isdir(os.path.join(root, cla))]
    # 排序，保证顺序一致
    flower_class.sort()
    # class_indices={文件夹名：索引}
    class_indices = dict((k, v) for v, k in enumerate(flower_class))
    json_str = json.dumps(dict((val, key) for key, val in class_indices.items()), indent=4)
    with open('class_indices.json', 'w') as json_file:
        json_file.write(json_str)

    train_images_path = []  # 存储训练集的所有图片路径
    train_images_label = []  # 存储训练集图片对应索引信息
    val_images_path = []  # 存储验证集的所有图片路径
    val_images_label = []  # 存储验证集图片对应索引信息
    every_class_num = []  # 存储每类的样本总数
    supported = [".jpg", ".JPG", ".png", ".PNG"]  # 支持的文件后缀类型
    # 遍历每个文件夹下的文件
    for cla in flower_class:
        cla_path = os.path.join(root, cla)
        # 遍历获取supported支持格式的所有文件路径
        images = [os.path.join(root, cla, i) for i in os.listdir(cla_path)
                  if os.path.splitext(i)[-1] in supported]
        # 获取该类别对应的索引
        image_class = class_indices[cla]
        # 记录该类别的样本数量
        every_class_num.append(len(images))
        # 按比例随机采样验证样本
        val_path = random.sample(images, k=int(len(images) * val_rate))

        for img_path in images:
            if img_path in val_path:  # 如果该路径在采样的验证集样本中则存入验证集
                val_images_path.append(img_path)
                val_images_label.append(image_class)
            else:  # 否则存入训练集
                train_images_path.append(img_path)
                train_images_label.append(image_class)

    print("{} images were found in the dataset.".format(sum(every_class_num)))
    print("{} images for training.".format(len(train_images_path)))
    print("{} images for validation.".format(len(val_images_path)))

    plot_image = False
    if plot_image:
        # 绘制每种类别个数柱状图
        plt.bar(range(len(flower_class)), every_class_num, align='center')
        # 将横坐标0,1,2,3,4替换为相应的类别名称
        plt.xticks(range(len(flower_class)), flower_class)
        # 在柱状图上添加数值标签
        for i, v in enumerate(every_class_num):
            plt.text(x=i, y=v + 5, s=str(v), ha='center')
        # 设置x坐标
        plt.xlabel('image class')
        # 设置y坐标
        plt.ylabel('number of images')
        # 设置柱状图的标题
        plt.title('flower class distribution')
        plt.show()

    with open('./biecaibaikuai/json/train_images_path.json', 'w') as json_file:
        json_file.write(json.dumps(train_images_path))
    with open('./biecaibaikuai/json/train_images_label.json', 'w') as json_file:
        json_file.write(json.dumps(train_images_label))
    with open('./biecaibaikuai/json/val_images_path.json', 'w') as json_file:
        json_file.write(json.dumps(val_images_path))
    with open('./biecaibaikuai/json/val_images_label.json', 'w') as json_file:
        json_file.write(json.dumps(val_images_label))

    return train_images_path, train_images_label, val_images_path, val_images_label

def train_one_epoch(model, optimizer, data_loader, device, epoch,tb_writer):
    model.train()
    loss_function = torch.nn.CrossEntropyLoss()
    accu_loss = torch.zeros(1).to(device)  # 累计损失
    accu_num = torch.zeros(1).to(device)   # 累计预测正确的样本数
    optimizer.zero_grad()

    sample_num = 0
    data_loader = tqdm(data_loader)
    for step, data in enumerate(data_loader):
        images, labels = data #(8,3,224,224) (8)
        sample_num += images.shape[0]
        pred = model(images.to(device))
        pred_classes = torch.max(pred, dim=1)[1]
        accu_num += torch.eq(pred_classes, labels.to(device)).sum()

        loss = loss_function(pred, labels.to(device))
        loss.backward()
        accu_loss += loss.detach()#取值 张量累加

        data_loader.desc = "[train epoch {}] loss: {:.3f}, acc: {:.3f}".format(epoch,
                                                                               accu_loss.item() / (step + 1),
                                                                               accu_num.item() / sample_num)

        if not torch.isfinite(loss):
            print('WARNING: non-finite loss, ending training ', loss)
            sys.exit(1)



        optimizer.step()
        optimizer.zero_grad()

    return accu_loss.item() / (step + 1), accu_num.item() / sample_num


@torch.no_grad()
def evaluate(model, data_loader, device, epoch):
    loss_function = torch.nn.CrossEntropyLoss()

    model.eval()

    accu_num = torch.zeros(1).to(device)   # 累计预测正确的样本数
    accu_loss = torch.zeros(1).to(device)  # 累计损失

    sample_num = 0
    data_loader = tqdm(data_loader)
    for step, data in enumerate(data_loader):
        images, labels = data
        sample_num += images.shape[0]

        pred = model(images.to(device))
        pred_classes = torch.max(pred, dim=1)[1]
        accu_num += torch.eq(pred_classes, labels.to(device)).sum()

        loss = loss_function(pred, labels.to(device))
        accu_loss += loss

        data_loader.desc = "[valid epoch {}] loss: {:.3f}, acc: {:.3f}".format(epoch,
                                                                               accu_loss.item() / (step + 1),
                                                                               accu_num.item() / sample_num)

    return accu_loss.item() / (step + 1), accu_num.item() / sample_num

def train(args, model, device, train_loader, optimizer, epoch):
    model.train()
    for batch_idx, (data, target) in enumerate(train_loader):# 938轮 = 6万/64
        #data：(64张,1通道,28,28)图片
        data, target = data.to(device), target.to(device) #target：64个数字是几的10分类标签
        optimizer.zero_grad() #梯度清零
        output = model(data) #前馈
        loss = F.nll_loss(output, target) #平均损失，默认reduction='mean'，batch求和(64)->除64
        loss.backward()      #反馈
        optimizer.step() #更新权重
        # plt.show()
        # plt.imshow(transforms.ToPILImage()(data[0]))
        # writer.add_graph(model,(data,))
        # if batch_idx == 0: writer.add_graph(model, (data,))
        if batch_idx % 8 == 0:
            print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                epoch, batch_idx * len(data), len(train_loader.dataset),
                100. * batch_idx / len(train_loader), loss.item()))
            # if args.dry_run:
            #     break
#output max索引位置一样表示匹配。谁规定的？ output[batch_index][target_index]
@torch.no_grad()
def test(model, device, test_loader, epoch):
    model.eval()
    test_loss = 0 #epoch求和
    correct = 0
    with torch.no_grad():
        for data, target in test_loader:   #10轮
            data, target = data.to(device), target.to(device)
            output = model(data)           #(1000,1,28,28) -> (1000,10)
            test_loss += F.nll_loss(output, target, reduction='sum').item() #batch求和(1000)
            #底下算手写公式
            pred = output.argmax(dim=1, keepdim=True)  #(1000,1) 最大概率的索引,非独热编码
            target1 = target.view_as(pred) #y跟y^对齐(1000)->(1000,1) 
            bool_tensor = pred.eq(target1) #y跟y^相等，转bool (1000,1)
            tnum = bool_tensor.sum()       #求和 false=0 true=1
            num = tnum.item()              #数字张量转数字
            correct += num

    test_loss /= len(test_loader.dataset) #平均损失=损失和/10000
   

    writer.add_scalar('test_loss', test_loss, epoch)
    writer.add_scalar('test_zql', 100. * correct / len(test_loader.dataset), epoch)
    writer.flush()

    print('\nTest set: Average loss: {:.4f}, Accuracy: {}/{} ({:.0f}%)\n'.format(
        test_loss, correct, len(test_loader.dataset),
        100. * correct / len(test_loader.dataset))) #正确率

    loss率 = test_loss * 100
    正确倒率 = 100 - 100. * correct / len(test_loader.dataset)
    率 = loss率 + 正确倒率
    
    return 率