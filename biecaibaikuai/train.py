import os
import math
import argparse

import torch
import torch.optim as optim
import torch.optim.lr_scheduler as lr_scheduler
from torch.utils.tensorboard import SummaryWriter
from torchvision import transforms,models
import torch.nn.functional as F
from torch import nn
from util import read_split_data,train_one_epoch,evaluate,train,test
from my_dataset import MyDataSet
class myResnet(nn.Module):
    def __init__(self, resnet):
        super(myResnet, self).__init__()
        self.resnet = resnet
        self.quanlianjie = nn.Linear(2048, 4)

    def forward(self, img, att_size=6):
        x = img

        x = self.resnet.conv1(x)
        x = self.resnet.bn1(x)
        x = self.resnet.relu(x)
        x = self.resnet.maxpool(x)

        x = self.resnet.layer1(x)
        x = self.resnet.layer2(x)
        x = self.resnet.layer3(x)
        x = self.resnet.layer4(x)

        fc = x.mean(3).mean(2).squeeze()
        # att = F.adaptive_avg_pool2d(x,[att_size,att_size]).squeeze().permute(1, 2, 0)

        result = self.quanlianjie(fc)
        result = F.log_softmax(result, dim=1)
        return result
def main(args):
    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    train_images_path, train_images_label, val_images_path, val_images_label = read_split_data('../baikuaimg')
    # print(train_images_path)
    data_transform = {
        "train": transforms.Compose([transforms.Resize(224),
                                    #  transforms.RandomHorizontalFlip(),#随机镜像翻转
                                     transforms.ToTensor(),#(0,255)->(0,1)
                                     transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])]),#(0,1)->(-1,1)
        "val": transforms.Compose([transforms.Resize(224),
                                #    transforms.CenterCrop(224),
                                   transforms.ToTensor(),
                                   transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])])}
        # 实例化训练数据集
    train_dataset = MyDataSet(images_path=train_images_path,
                              images_class=train_images_label,
                              transform=data_transform["train"])

    # 实例化验证数据集
    val_dataset = MyDataSet(images_path=val_images_path,
                            images_class=val_images_label,
                            transform=data_transform["val"])

    batch_size = args.batch_size
    nw = min([os.cpu_count(), batch_size if batch_size > 1 else 0, 8])  # number of workers
    print('{} 个子进程'.format(nw))
    train_loader = torch.utils.data.DataLoader(train_dataset,
                                               batch_size=batch_size,
                                               shuffle=True,#每轮epoch洗牌
                                               pin_memory=True,
                                               num_workers=nw,
                                               collate_fn=train_dataset.collate_fn)

    val_loader = torch.utils.data.DataLoader(val_dataset,
                                             batch_size=batch_size,
                                             shuffle=False,
                                             pin_memory=True,
                                             num_workers=nw,
                                             collate_fn=val_dataset.collate_fn)
    #加载网络，但不使用原参数，1000分类改为5分类
    # resnet101 = models.resnet101(pretrained=True)
    # model = resnet101(pretrainen=False, num_classes=5).to(device)
    #加载参数
    # model.load_state_dict(torch.load('x.pth', map_location=device))

    resnet101=models.resnet101(pretrained=True).eval()
    model=myResnet(resnet101).cuda(device)
    if (os.path.exists(args.weights)):
        model.load_state_dict(torch.load(args.weights, map_location=device), strict=False)
        print('继续训练')
    pg = [p for p in model.parameters() if p.requires_grad]
    optimizer = optim.SGD(pg, lr=args.lr, momentum=0.9, weight_decay=5E-5)
    lf = lambda x: ((1 + math.cos(x * math.pi / args.epochs)) / 2) * (1 - args.lrf) + args.lrf  # cosine
    scheduler = lr_scheduler.LambdaLR(optimizer, lr_lambda=lf) #将每个参数组的学习率设置为初始 lr 乘以给定函数
    
    初始率 = 10
    for epoch in range(args.epochs):
        # # train
        # train_loss, train_acc = train_one_epoch(model=model,
        #                                         optimizer=optimizer,
        #                                         data_loader=train_loader,
        #                                         device=device,
        #                                         epoch=epoch, tb_writer=0)
        # scheduler.step()
        # # validate
        # val_loss, val_acc = evaluate(model=model,
        #                              data_loader=val_loader,
        #                              device=device,
        #                              epoch=epoch)
        train(args, model, device, train_loader, optimizer, epoch)
        率 = test(model, device, val_loader, epoch)
        scheduler.step()
        if (率 < 初始率):
            torch.save(model.state_dict(), args.weights)
            初始率 = 率
            print('存一波助助兴', 率)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--num_classes', type=int, default=5)
    parser.add_argument('--epochs', type=int, default=100)
    parser.add_argument('--batch-size', type=int, default=8)
    parser.add_argument('--lr', type=float, default=0.001)
    parser.add_argument('--lrf', type=float, default=0.01)

    # 数据集所在根目录 花
    # http://download.tensorflow.org/example_images/flower_photos.tgz
    parser.add_argument('--data-path', type=str,
                        default="./data/flower_photos")
    parser.add_argument('--model-name', default='', help='create model name')

    # 预训练权重路径，如果不想载入就设置为空字符 
    parser.add_argument('--weights', type=str, default='./cundang.pt',
                        help='initial weights path')
    # 是否冻结权重
    parser.add_argument('--freeze-layers', type=bool, default=True)
    parser.add_argument('--device', default='cuda:0', help='device id (i.e. 0 or 0,1 or cpu)')

    opt = parser.parse_args()

    main(opt)

