import os
import json
import torch
import torch.nn as nn
from torchvision import transforms, datasets, utils
import matplotlib.pyplot as plt
import numpy as np
import torch.optim as optim
from tqdm import tqdm
from model import AlexNet


def main():
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print("using {} device.".format(device))
    data_transform = {
        # 该方法为数据预处理方法
        # 当关键字为train时，返回训练集的数据与处理方法
        "train": transforms.Compose([transforms.RandomResizedCrop(224),  # 将图片用随机裁剪方法裁剪成224*224
                                     transforms.RandomHorizontalFlip(),  # 在水平方向随机翻转
                                     transforms.ToTensor(),  # 将它转化成Tensor
                                     transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))]),
        # 将数据进行标准化处理
        # 当关键字为val时，返回训练集的数据与处理方法
        "val": transforms.Compose([transforms.Resize((224, 224)),  # 将图片转化成224*224的大小
                                   transforms.ToTensor(),  # 将数据转化成tensor
                                   transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))]),
        # 将数据进行标准化处理
        "test": transforms.Compose([transforms.Resize((224, 224)),
                                    transforms.ToTensor(),
                                    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])
    }
    data_root = r'S:/pycharm/IDcar/data/'
    train_dataset = datasets.ImageFolder(data_root + 'train', data_transform['train'])
    val_dataset = datasets.ImageFolder(data_root + 'val', data_transform['val'])
    test_dataset = datasets.ImageFolder(data_root + 'test', data_transform['test'])

    train_num = len(train_dataset)
    val_num = len(val_dataset)
    test_num = len(test_dataset)

    # {'daisy':0, 'dandelion':1, 'roses':2, 'sunflower':3, 'tulips':4}
    flower_list = train_dataset.class_to_idx  # 获取分类名称所对应的索引
    cla_dict = dict((val, key) for key, val in flower_list.items())

    json_str = json.dumps(cla_dict, indent=4)
    with open('class_indices.json', 'w') as json_file:
        json_file.write(json_str)

    batch_size = 32  # 定义batch_size=32
    nw = min([os.cpu_count(), batch_size if batch_size > 1 else 0, 8])  # number of workers
    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=nw)
    # train_loader函数是为了随机在数据集中获取一批批数据，num_workers=0加载数据的线程个数，在windows系统下该数为                 0，意思为在windows系统下使用一个主线程加载数据
    validate_loader = torch.utils.data.DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=nw)
    test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=nw)

    print("using {} images for training, {} images for validation, and {} images for testing.".format(train_num, val_num, test_num))

    net = AlexNet(num_classes=len(flower_list), init_weights=True).to(device)  # num_classes=4有4种类别，初始化权重
    loss_function = nn.CrossEntropyLoss()  # 定义损失函数，针对多类别的损失交叉熵函数
    # pat = list(net.parameters())
    optimizer = optim.Adam(net.parameters(), lr=0.0002)

    epochs = 3
    save_path = './AlexNet.pth'  # 保存准确率最高的那次模型的路径
    best_acc = 0.0  # 最佳准确率
    # train_steps = len(train_loader)

    results_file = 'training_results.txt'
    with open(results_file, 'w') as f:
        f.write(f'Epoch, Train loss, Validation Accuracy, Test Accuracy,test_accurate\n')

    for epoch in range(epochs):  # train
        net.train()  # 使用net.train()方法，该方法中有dropout
        running_loss = 0.0  # 使用running_loss方法统计训练过程中的平均损失
        train_bar = tqdm(train_loader)

        for step, data in enumerate(train_bar):  # 遍历数据集
            images, labels = data  # 将数据分为图像标签
            optimizer.zero_grad()  # 清空之前的梯度信息
            outputs = net(images.to(device))  # 通过正向传播的到输出
            loss = loss_function(outputs, labels.to(device))  # 指定设备gpu或者cpu,通过Loss_function函数计算预测值与真实值之间的差距
            loss.backward()  # 将损失反向传播到每一个节点
            optimizer.step()  # 通过optimizer更新每一个参数
            # print statistics
            running_loss += loss.item()  # 累加损失
            # print train process
            train_bar.desc = f'Epoch [{epoch + 1}/{epochs}] Loss: {loss.item():.4f}'

        avg_loss = running_loss / len(train_loader)
        print(f'Epoch [{epoch + 1}/{epochs}] Loss: {avg_loss:.4f}]')

        net.eval()  # 预测过程中使用net.eval()函数，该函数会关闭掉dropout
        acc = 0.0  # accumulate accurate number / epoch
        val_running_loss = 0.0

        with torch.no_grad():  # 使用该函数，禁止pytorch对参数进行跟踪，即训练过程中不会计算损失梯度
            val_bar = tqdm(validate_loader)
            for val_data in val_bar:  # 遍历验证集
                val_images, val_labels = val_data  # 将数据划分为图片和标签
                outputs = net(val_images.to(device))
                loss = loss_function(outputs, val_labels.to(device))  # 计算验证损失值
                val_running_loss += loss.item()  # 累积验证损失

                predict_y = torch.max(outputs, dim=1)[1]  # 求的预测过程中最有可能的标签
                acc += torch.eq(predict_y, val_labels.to(device)).sum().item()  # 准确的个数累加

        val_accurate = acc / val_num  # 计算验证集的正确率
        avg_val_loss = val_running_loss / val_num
        print(
            f'Validation Accuracy of the model on the validation images: {val_accurate:.4f}, Validation Loss: {avg_val_loss:.4f}')

        # 保存最佳模型,如果当前准确率大于历史最优准确率，就将当前的准确率赋给最优准确率，并将参数进行保存
        if val_accurate > best_acc:
            best_acc = val_accurate
            torch.save(net.state_dict(), save_path)  # 保存模型

        # 测试过程
        # net.load_state_dict(torch.load(save_path))  # 加载最佳模型
        net.eval()
        test_acc = 0.0
        with torch.no_grad():
            test_bar = tqdm(test_loader)
            for test_data in test_bar:
                test_images, test_labels = test_data
                outputs = net(test_images.to(device))
                predict_y = torch.max(outputs, dim=1)[1]
                test_acc += torch.eq(predict_y, test_labels.to(device)).sum().item()

        test_accurate = test_acc / test_num  # 计算测试集的正确率
        print(f'Test Accuracy of the model on the test images: {test_accurate:.4f}')

        # 将结果写入文件
        with open(results_file, 'a') as f:
            f.write(f'Epoch:{epoch + 1},Train loss:{avg_loss:.4f},Validation Accuracy:{val_accurate:.4f},Test Loss:{avg_val_loss:.4f},test_Accuracy:{test_accurate:.4f}\n')

    print('Finished Training')


if __name__ == '__main__':

    main()
