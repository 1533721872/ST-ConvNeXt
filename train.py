import os
import argparse

import torch
import torch.optim as optim
from torch.utils.tensorboard import SummaryWriter
from torchvision import transforms
import torchvision.models as models
from my_dataset import MyDataSet
from model import convnext_tiny as create_model
from utils import *
import warnings
warnings.filterwarnings('ignore')
import torch.quantization
#from conv_change import *

#用于训练和评估
def main(args):
    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    print(f"using {device} device.")

    if os.path.exists("/share/home/u2020060/Text1/ST-ConvNeXt/weights") is False:
        os.makedirs("/share/home/u2020060/Text1/ST-ConvNeXt/weights")

    tb_writer = SummaryWriter('/share/home/u2020060/Text1/ST-ConvNeXt/log') #创建一个 SummaryWriter 对象，用于记录训练过程中的指标和可视化。
    #调用函数 read_split_data，从指定的数据路径中读取训练图像和验证图像的路径和标签信息。
    train_images_path, train_images_label, val_images_path, val_images_label = read_split_data(args.data_path)
    img_size = 256
    data_transform = {
        "train": transforms.Compose([transforms.RandomResizedCrop(img_size),#随机裁剪
                                     transforms.RandomRotation(45),
                                     transforms.RandomHorizontalFlip(), #水平翻转
                                     transforms.RandomVerticalFlip(),
                                     transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.05),
                                     transforms.ToTensor(),
                                     transforms.Normalize([0.485, 0.490, 0.50], [0.218, 0.201, 0.197])]),
        "val": transforms.Compose([
                                   #transforms.Resize(int(img_size * 1.143)),
                                   transforms.CenterCrop(img_size),
                                   transforms.ToTensor(),
                                   transforms.Normalize([0.485, 0.490, 0.50], [0.218, 0.201, 0.197])])}

    # 实例化训练数据集
    train_dataset = MyDataSet(images_path=train_images_path,
                              images_class=train_images_label,
                              transform=data_transform["train"])

    # 实例化验证数据集
    val_dataset = MyDataSet(images_path=val_images_path,
                            images_class=val_images_label,
                            transform=data_transform["val"])

    batch_size = args.batch_size #获取批量大小
    nw = min([os.cpu_count(), batch_size if batch_size > 1 else 0, 8])  # 多进程数
    print('Using {} dataloader workers every process'.format(nw))
    train_loader = torch.utils.data.DataLoader(train_dataset,
                                               batch_size=batch_size,
                                               shuffle=True,
                                               pin_memory=True,
                                               num_workers=nw,
                                               collate_fn=train_dataset.collate_fn)

    val_loader = torch.utils.data.DataLoader(val_dataset,
                                             batch_size=batch_size,
                                             shuffle=False,
                                             pin_memory=True,
                                             num_workers=nw,
                                             collate_fn=val_dataset.collate_fn)

    model= create_model(num_classes=args.num_classes).to(device)
    # if args.weights != "":
    #     assert os.path.exists(args.weights), "weights file: '{}' not exist.".format(args.weights)
    #     weights_dict = torch.load(args.weights, map_location=device)["model"]
    #     # 删除有关分类类别的权重，排除与当前任务不相关的权重。
    #     for k in list(weights_dict.keys()):
    #         if "head" in k:
    #             del weights_dict[k]
    #     print(model.load_state_dict(weights_dict, strict=False))
    if args.weight != "":
        assert os.path.exists(args.weight), "weights file: '{}' not exist.".format(args.weight)
        # 加载权重文件
        checkpoint = torch.load(args.weight, map_location=device)
        # 打印权重文件的键，检查结构
        #print("Checkpoint keys:", checkpoint.keys())

        # 尝试访问不同的键，看看哪个是正确的
        if "model" in checkpoint:
            weights_dict = checkpoint["model"]
        elif "state_dict" in checkpoint:
            weights_dict = checkpoint["state_dict"]
        else:
            weights_dict = checkpoint  # 如果没有找到特定键，假设整个checkpoint就是权重字典

        # 删除有关分类类别的权重，排除与当前任务不相关的权重
        for k in list(weights_dict.keys()):
            if "head" in k:
                del weights_dict[k]

        #print(model.load_state_dict(weights_dict, strict=False))
    if args.freeze_layers:
        for name, para in model.named_parameters():
            # 除head外，其他权重全部冻结
            if "head" not in name:
                para.requires_grad_(False)
            else:
                print("training {}".format(name))

    pg = [p for p in model.parameters() if p.requires_grad]
    pg = get_params_groups(model, weight_decay=args.wd)
    optimizer = optim.AdamW(pg, lr=args.lr, weight_decay=args.wd)

    lr_scheduler = create_lr_scheduler(optimizer, len(train_loader), args.epochs,
                                       warmup=True, warmup_epochs=1)

    best_acc = 0.
    for epoch in range(args.epochs):
        # 计算训练集上的损失和准确率
        train_loss, train_acc = train_one_epoch(model=model,
                                                optimizer=optimizer,
                                                data_loader=train_loader,
                                                device=device,
                                                epoch=epoch,
                                                lr_scheduler=lr_scheduler)

        # 计算验证集上的损失和准确率
        val_loss, val_acc,confusion_matrix = evaluate(model=model,
                                     data_loader=val_loader,
                                     device=device,
                                     epoch=epoch)

        tags = ["train_loss", "train_acc", "val_loss", "val_acc", "learning_rate"]
        tb_writer.add_scalar(tags[0], train_loss, epoch)
        tb_writer.add_scalar(tags[1], train_acc, epoch)
        tb_writer.add_scalar(tags[2], val_loss, epoch)
        tb_writer.add_scalar(tags[3], val_acc, epoch)
        tb_writer.add_scalar(tags[4], optimizer.param_groups[0]["lr"], epoch)

        if best_acc < val_acc:
            best_confusion_matrix = confusion_matrix
            torch.save(model.state_dict(), "/share/home/u2020060/Text1/ST-ConvNeXt/weights/best_model.pth")
            best_acc = val_acc

    _, _, confusion_matrix = evaluate(model=model,
                                      data_loader=val_loader,
                                      device=device,
                                      epoch=epoch)
    plot_confusion_matrix(best_confusion_matrix,class_names)
if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--num_classes', type=int, default=21)
    parser.add_argument('--epochs', type=int, default=250)
    parser.add_argument('--batch-size', type=int, default=8)
    parser.add_argument('--lr', type=float, default=5e-4)
    parser.add_argument('--wd', type=float, default=5e-2) # 权重衰减系数

    parser.add_argument('--data-path', type=str,
                        default="/share/home/u2020060/Text1/ST-ConvNeXt/data/UCMerced")

    parser.add_argument('--weight', type=str, default='/share/home/u2020060/Text1/ST-ConvNeXt/weight/STConvNeXt.pth',
                        help='initial weights path')


    # 是否冻结head以外所有权重
    parser.add_argument('--freeze-layers', type=bool, default=False)
    parser.add_argument('--device', default='cuda:0', help='device id (i.e. 0 or 0,1 or cpu)')

    opt = parser.parse_args()

    main(opt)
