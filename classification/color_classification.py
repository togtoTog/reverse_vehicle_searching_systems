import torch
import torch.nn as nn
import torch.optim as optim

import torchvision.transforms as transforms
from torchvision.datasets import ImageFolder
from torchvision import models

import os
import time

import numpy as np
from PIL import Image
import matplotlib.pyplot as plt

from config import color_cfg


def load_dataset():
    # 设置训练集
    train_transform = transforms.Compose([
        transforms.RandomResizedCrop(224),  # 随机裁剪到224x224大小
        transforms.RandomHorizontalFlip(),  # 随机水平翻转
        transforms.RandomRotation(degrees=15),  # 随机旋转
        transforms.ToTensor(),  # 转化成Tensor
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[
            0.229, 0.224, 0.225])  # 正则化
    ])
    train_dataset = ImageFolder(color_cfg.TRAIN.DATASET,
                                transform=train_transform)  # 训练集数据
    train_loader = torch.utils.data.DataLoader(dataset=train_dataset, batch_size=color_cfg.TRAIN.BATCH_SIZE, shuffle=True,
                                               num_workers=2)  # 加载数据

    # 设置测试集
    test_transform = transforms.Compose([
        transforms.Resize((224, 224)),  # resize到224x224大小
        transforms.ToTensor(),  # 转化成Tensor
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[
            0.229, 0.224, 0.225])  # 正则化
    ])
    test_dataset = ImageFolder(color_cfg.TEST.DATASET,
                               transform=test_transform)  # 测试集数据
    test_loader = torch.utils.data.DataLoader(dataset=test_dataset, batch_size=color_cfg.TRAIN.BATCH_SIZE, shuffle=False,
                                              num_workers=2)  # 加载数据

    # 打印类别内容
    class_names = test_dataset.classes
    print(class_names)

    # 打印训练集、测试集大小
    train_data_size = len(train_dataset)
    valid_data_size = len(test_dataset)
    print(train_data_size, valid_data_size)

    return train_loader, test_loader, train_data_size, valid_data_size


def train_init():
    # 设置模型结构
    model = models.resnet34(pretrained=False).to(color_cfg.MODEL.DEVICE)
    # 加载预训练模型参数
    model.load_state_dict(torch.load(color_cfg.MODEL.PRETRAIN))

    # 设置损失函数
    loss_function = nn.CrossEntropyLoss()
    # 设置优化器和学习率
    optimizer = optim.Adam(model.parameters(), lr=0.0002)
    return model, loss_function, optimizer

# 定义训练函数


def train_and_valid(train_loader, test_loader, train_data_size, valid_data_size, model, loss_function, optimizer,
                    epochs=30):
    # device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    device = torch.device(color_cfg.MODEL.DEVICE)
    history = []
    best_acc = 0.0
    best_epoch = 0

    for epoch in range(epochs):
        print('model_{}.pth'.format(epoch + 1))
        epoch_start = time.time()
        print("Epoch: {}/{}".format(epoch + 1, epochs))

        model.train()

        train_loss = 0.0
        train_acc = 0.0
        valid_loss = 0.0
        valid_acc = 0.0

        for i, (inputs, labels) in enumerate(train_loader):
            inputs = inputs.to(device)
            labels = labels.to(device)

            # 因为这里梯度是累加的，所以每次记得清零
            optimizer.zero_grad()

            outputs = model(inputs)

            loss = loss_function(outputs, labels)

            loss.backward()

            optimizer.step()

            train_loss += loss.item() * inputs.size(0)

            ret, predictions = torch.max(outputs.data, 1)
            correct_counts = predictions.eq(labels.data.view_as(predictions))

            acc = torch.mean(correct_counts.type(torch.FloatTensor))

            train_acc += acc.item() * inputs.size(0)

        with torch.no_grad():
            model.eval()

            for j, (inputs, labels) in enumerate(test_loader):
                inputs = inputs.to(device)
                labels = labels.to(device)

                outputs = model(inputs)

                loss = loss_function(outputs, labels)

                valid_loss += loss.item() * inputs.size(0)

                ret, predictions = torch.max(outputs.data, 1)
                correct_counts = predictions.eq(
                    labels.data.view_as(predictions))

                acc = torch.mean(correct_counts.type(torch.FloatTensor))

                valid_acc += acc.item() * inputs.size(0)

        avg_train_loss = train_loss / train_data_size
        avg_train_acc = train_acc / train_data_size

        avg_valid_loss = valid_loss / valid_data_size
        avg_valid_acc = valid_acc / valid_data_size
        # 将每一轮的损失值和准确率记录下来
        history.append([avg_train_loss, avg_valid_loss,
                       avg_train_acc, avg_valid_acc])

        if best_acc < avg_valid_acc:
            best_acc = avg_valid_acc
            best_epoch = epoch + 1

        epoch_end = time.time()
        # 打印每一轮的损失值和准确率，效果最佳的验证集准确率
        print(
            "Epoch: {:03d}, Training: Loss: {:.4f}, Accuracy: {:.4f}%, \n\t\tValidation: Loss: {:.4f}, Accuracy: {:.4f}%, Time: {:.4f}s".format(
                epoch + 1, avg_valid_loss, avg_train_acc * 100, avg_valid_loss, avg_valid_acc * 100, epoch_end - epoch_start
            ))
        print("Best Accuracy for validation : {:.4f} at epoch {:03d}".format(
            best_acc, best_epoch))
        torch.save(model.state_dict(), os.path.join(
            color_cfg.MODEL.CHECKPOINT, f'model_{epoch}.pth'))

    return history


if __name__ == '__main__':
    train_loader, test_loader, train_data_size, valid_data_size = load_dataset()
    model, loss_function, optimizer = train_init()
    history = train_and_valid(train_loader, test_loader, train_data_size, valid_data_size,
                    model, loss_function, optimizer, color_cfg.TRAIN.EPOCHS)
    # 将训练参数用图表示出来
    history = np.array(history)
    plt.plot(history[:, 0:2])
    plt.legend(['Tr Loss', 'Val Loss'])
    plt.xlabel('Epoch Number')
    plt.ylabel('Loss')
    plt.ylim(0, 1.1)
    plt.savefig(color_cfg.METRIC.LOSS_PIC)
    # plt.show()

    plt.cla()

    plt.plot(history[:, 2:4])
    plt.legend(['Tr Accuracy', 'Val Accuracy'])
    plt.xlabel('Epoch Number')
    plt.ylabel('Accuracy')
    plt.ylim(0, 1.1)
    plt.savefig(color_cfg.METRIC.ACCURACY_PIC)
