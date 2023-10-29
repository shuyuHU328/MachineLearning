import torch
import torch.nn as nn
import torchvision.datasets as datasets
import torch.utils.data
import torchvision.transforms as transforms
import matplotlib.pyplot as plt


class LeNet5(nn.Module):
    def __init__(self):
        super(LeNet5, self).__init__()
        # 卷积层C1
        self.conv1 = nn.Conv2d(in_channels=3, out_channels=6, kernel_size=5)
        # 采样层S2
        self.pool1 = nn.MaxPool2d(kernel_size=2, stride=2)
        # 卷积层C3
        self.conv2 = nn.Conv2d(in_channels=6, out_channels=16, kernel_size=5)
        # 采样层S4
        self.pool2 = nn.MaxPool2d(kernel_size=2, stride=2)
        # 全连接层C5
        self.fc1 = nn.Linear(in_features=16 * 5 * 5, out_features=120)
        # 全连接层F6
        self.fc2 = nn.Linear(in_features=120, out_features=84)
        # 输出层
        self.fc3 = nn.Linear(in_features=84, out_features=10)

    def forward(self, x):
        x = self.pool1(torch.relu(self.conv1(x)))
        x = self.pool2(torch.relu(self.conv2(x)))
        x = x.view(-1, 16 * 5 * 5)
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        x = self.fc3(x)
        return x


# 定义超参数
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = LeNet5().to(device)
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.00065, weight_decay=0.0008)
classes = ('plane', 'car', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck')
epochs = 60

if __name__ == '__main__':
    transform = transforms.Compose([transforms.Resize((32, 32)),
                                    transforms.RandomHorizontalFlip(p=0.45),  # 随机水平翻转
                                    transforms.ToTensor(),  # 标准化
                                    transforms.Normalize((0.4914, 0.4822, 0.4465),
                                                         (0.2023, 0.1994, 0.2010))])  # 根据数据做出调整
    # transform_train = transforms.Compose([transforms.Resize((32, 32)),
    #                                       transforms.RandomHorizontalFlip(p=0.45),  # 随机水平翻转
    #                                       transforms.ToTensor(),  # 标准化
    #                                       transforms.Normalize((0.48836562, 0.48134598, 0.4451678),
    #                                                            (0.24833508, 0.24547848, 0.26617324))])
    # transform_test = transforms.Compose([transforms.Resize((32, 32)),
    #                                      transforms.ToTensor(),  # 标准化
    #                                      transforms.Normalize((0.47375134, 0.47303376, 0.42989072),
    #                                                           (0.25467148, 0.25240466, 0.26900575))])
    # 读取CIFAR-10数据集
    train_dataset = datasets.CIFAR10(root='./data/', train=True, transform=transform, download=True)
    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=32, shuffle=True, num_workers=4)
    test_dataset = datasets.CIFAR10(root='./data/', train=False, transform=transform, download=True)
    test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=32, shuffle=False, num_workers=2)
    data = []
    for epoch in range(epochs):
        model.train()
        for images, labels in train_loader:
            # 将数据放入模型
            images = images.to(device)
            labels = labels.to(device)
            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
        # 在测试集上测试模型
        if epoch < 55:
            continue
        model.eval()
        correct = 0
        with torch.no_grad():
            for images, labels in test_loader:
                # 将数据放入模型
                images = images.to(device)
                labels = labels.to(device)
                outputs = model(images)
                _, predicted = torch.max(outputs, 1)
                correct += (predicted == labels).sum().item()
        # 输出该epoch的精度
        accuracy = 100 * correct / len(test_dataset)
        data.append(accuracy)
        print('Epoch [{}/{}], Loss: {:.4f}, Accuracy: {:.2f}%'.format(epoch + 1, epochs, loss.item(), accuracy))
    # 打印10个分类的准确率
    class_correct = list(0. for i in range(10))
    class_total = list(0. for i in range(10))
    with torch.no_grad():
        for images, labels in test_loader:
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)  # outputs的维度是：4*10
            # torch.max(outputs.data, 1)返回outputs每一行中最大值的那个元素，且返回其索引
            # 此时predicted的维度是：4*1
            _, predicted = torch.max(outputs, 1)
            # 此时c的维度：4将预测值与实际标签进行比较，且进行降维
            c = (predicted == labels).squeeze()
            for i in range(4):
                label = labels[i]
                class_correct[label] += c[i].item()
                class_total[label] += 1
    for i in range(10):
        print('Accuracy of %5s : %.2f %%' % (classes[i], 100 * class_correct[i] / class_total[i]))
    # 保存模型
    torch.save(model.state_dict(), 'model.pth')
