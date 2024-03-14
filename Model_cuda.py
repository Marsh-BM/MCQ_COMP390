# 导入必要的库
import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms
from torch.utils.data import DataLoader, random_split
import torch.nn.functional as F
import time
import matplotlib.pyplot as plt
from matplotlib.ticker import FormatStrFormatter
import numpy as np


# 这里得bug有可能导致环境崩溃！！！！并不是一个好的解决方法
import os
os.environ['KMP_DUPLICATE_LIB_OK']='TRUE'




# 定义神经网络模型
class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        # 定义卷积层和全连接层
        self.conv1 = nn.Conv2d(3, 32, kernel_size=3, padding=1)  # 第一个卷积层
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, padding=1)  # 第二个卷积层
        self.conv3 = nn.Conv2d(64, 128, kernel_size=3, padding=1)  # 第三个卷积层
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)  # 最大池化层，用于降维
        self.fc1 = nn.Linear(128 * 3 * 18, 512)  # 第一个全连接层，数字需要根据前面层的输出调整
        self.fc2 = nn.Linear(512, 7)  # 第二个全连接层，输出7个类别
        self.dropout = nn.Dropout(0.5)  # Dropout层，用于减少过拟合

    def forward(self, x):
        # 定义前向传播过程
        x = self.pool(F.relu(self.conv1(x)))  # 应用第一个卷积层+激活函数+池化
        x = self.pool(F.relu(self.conv2(x)))  # 应用第二个卷积层+激活函数+池化
        x = self.pool(F.relu(self.conv3(x)))  # 应用第三个卷积层+激活函数+池化
        x = x.view(-1, 128 * 3 * 18)  # 展平特征图
        x = F.relu(self.fc1(x))  # 应用第一个全连接层+激活函数
        x = self.dropout(x)  # 应用Dropout
        x = self.fc2(x)  # 应用第二个全连接层，得到最终的输出
        return x



# 训练和验证模型
def train_model(num_epochs, train_loader, val_loader, model, criterion, optimizer, model_path='COMP390Test1'):
    # 初始化记录
    train_losses = []
    val_losses = []
    train_accuracies = []
    val_accuracies = []

    for epoch in range(num_epochs):
        model.train()
        running_loss = 0.0
        correct = 0
        total = 0

        for images, labels in train_loader:
            images, labels = images.to(device), labels.to(device)

            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            running_loss += loss.item() * images.size(0)
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

        epoch_loss = running_loss / len(train_loader.dataset)
        epoch_acc = correct / total

        train_losses.append(epoch_loss)
        train_accuracies.append(epoch_acc)

        # 验证阶段
        model.eval()
        val_running_loss = 0.0
        val_correct = 0
        val_total = 0
        with torch.no_grad():
            for images, labels in val_loader:
                images, labels = images.to(device), labels.to(device)
                outputs = model(images)
                loss = criterion(outputs, labels)
                val_running_loss += loss.item() * images.size(0)
                _, predicted = torch.max(outputs.data, 1)
                val_total += labels.size(0)
                val_correct += (predicted == labels).sum().item()

        val_epoch_loss = val_running_loss / len(val_loader.dataset)
        val_epoch_acc = val_correct / val_total

        val_losses.append(val_epoch_loss)
        val_accuracies.append(val_epoch_acc)

        print(f'Epoch {epoch + 1}/{num_epochs}, Train Loss: {epoch_loss:.4f}, Train Accuracy: {epoch_acc:.4f}, Validation Loss: {val_epoch_loss:.4f}, Validation Accuracy: {val_epoch_acc:.4f}')

    # 调用plot_metrics函数在每个epoch结束后绘制指标
    plot_metrics(train_losses, val_losses, train_accuracies, val_accuracies,'Result/training_metrics.png')

    # 保存模型
    torch.save(model.state_dict(), model_path)
    print(f'Model saved to {model_path}')

   

# 测试模型的函数
# def test_model(test_loader, model):
#     model.eval()  # 设置模型为评估模式
#     correct = 0
#     total = 0
#     with torch.no_grad():  # 不计算梯度，加速推理
#         for images, labels in test_loader:
#             images, labels = images.to(device), labels.to(device)  # 将测试数据移动到正确的设备
#             outputs = model(images)
#             _, predicted = torch.max(outputs.data, 1)
#             total += labels.size(0)
#             correct += (predicted == labels).sum().item()
#     test_accuracy = correct / total
#     print(f'Test Accuracy: {test_accuracy:.4f}')

# def test_model(test_loader, model, device):
#     model.eval()  # 设置模型为评估模式
#     correct = 0
#     total = 0
#     misclassified = []  # 用于记录误分类的信息

#     with torch.no_grad():  # 不计算梯度，加速推理
#         for batch_idx, (images, labels, filenames) in enumerate(test_loader):
#             images, labels = images.to(device), labels.to(device)
#             outputs = model(images)
#             _, predicted = torch.max(outputs.data, 1)
#             total += labels.size(0)
#             correct += (predicted == labels).sum().item()

#             # 检查哪些图片分类错误了
#             mismatches = (predicted != labels)
#             for idx in np.where(mismatches.cpu().numpy())[0]:  # 从mismatches张量中提取索引
#                 misclassified.append((filenames[idx], labels[idx].item(), predicted[idx].item()))

#     test_accuracy = correct / total
#     print(f'Test Accuracy: {test_accuracy:.4f}')
#     # 如果需要，打印出分类错误的信息
#     if misclassified:
#         print("Misclassified images: [filename: (true label, predicted label)]")
#         for item in misclassified:
#             print(f"Filename: {item[0]} - (True: {item[1]}, Pred: {item[2]})")

#     return test_accuracy, misclassified

def test_model(test_loader, model, device):
    model.eval()  # 设置模型为评估模式
    correct = 0
    total = 0
    misclassified = []  # 用于记录误分类的信息

    with torch.no_grad():  # 不计算梯度，加速推理
        for batch_idx, (images, labels) in enumerate(test_loader):
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

            # 检查哪些图片分类错误了
            mismatches = (predicted != labels)
            mis_idxs = np.where(mismatches.cpu().numpy())[0]  # 从mismatches张量中提取索引
            for idx in mis_idxs:
                img_path = test_loader.dataset.samples[batch_idx * test_loader.batch_size + idx][0]  # 获取图片路径
                img_name = img_path.split('\\')[-1]  # 针对Windows路径
                class_label = test_loader.dataset.classes[labels[idx].item()]
                predicted_label = test_loader.dataset.classes[predicted[idx].item()]
                misclassified.append((img_name, class_label, predicted_label))

    test_accuracy = correct / total
    print(f'Test Accuracy: {test_accuracy:.4f}')
    # 如果需要，打印出分类错误的信息
    if misclassified:
        print("Misclassified images: [name: (true label, predicted label)]")
        for item in misclassified:
            print(f"Name {item[0]}: (True: {item[1]}, Pred: {item[2]})")

    return test_accuracy, misclassified



def plot_metrics(train_loss, val_loss, train_acc, val_acc, save_path):
    epochs = range(1, len(train_loss) + 1)
    plt.figure(figsize=(12, 4))
    
    # 绘制损失图
    plt.subplot(1, 2, 1)
    plt.plot(epochs, train_loss, label='Training Loss')
    plt.plot(epochs, val_loss, label='Validation Loss')
    plt.title('Training and Validation Loss')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.legend()
    plt.gca().yaxis.set_major_formatter(FormatStrFormatter('%.4f'))  # 设置y轴标签的格式
    plt.tight_layout()
    
    # 绘制准确率图
    plt.subplot(1, 2, 2)
    plt.plot(epochs, train_acc, label='Training Accuracy')
    plt.plot(epochs, val_acc, label='Validation Accuracy')
    plt.title('Training and Validation Accuracy')
    plt.xlabel('Epochs')
    plt.ylabel('Accuracy')
    plt.legend()
    plt.gca().yaxis.set_major_formatter(FormatStrFormatter('%.2f'))  # 设置y轴标签的格式
    plt.tight_layout()
    
    # 在显示之前保存图像
    plt.savefig(save_path, dpi=300)  # 保存图像到指定路径，dpi设置图像分辨率
    plt.show()



# 设定数据集路径（这需要根据实际情况进行替换）
dataset_path = 'Questions'
# 确保随机划分数据集的可复现性
torch.manual_seed(0)  # 设置随机种子

# 【检查CUDA是否可用，并定义设备】
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using {device} device")


# 图像预处理和增强 - 定义了一个转换流程，用于对图像数据进行处理和增强
data_transforms = transforms.Compose([
    transforms.Resize((30, 150)),  # 将图像大小统一调整为150x150
    # transforms.RandomHorizontalFlip(),  # 随机进行水平翻转，以增加数据多样性
    transforms.RandomRotation(20),  # 随机旋转图像，角度在-20到20度之间
    # transforms.RandomAffine(degrees=0,scale=(0.8, 1.2)),# 随机仿射变换，提高模型对于位置、缩放等因素的鲁棒性
    
    transforms.ToTensor(),  # 将图像转换为PyTorch张量
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    # 标准化图像，使用的均值和标准差是在ImageNet数据集上预先计算得到的
])

# 加载数据集，并根据比例划分为训练集和验证集
full_dataset = datasets.ImageFolder(root=dataset_path, transform=data_transforms)

# 划分数据集
train_size = int(0.8 * len(full_dataset))  # 80%的数据作为训练集
val_size = len(full_dataset) - train_size  # 剩余20%的数据作为验证集
train_dataset, val_dataset = random_split(full_dataset, [train_size, val_size], generator=torch.Generator().manual_seed(0))  # 使用相同的随机种子
# 创建数据加载器，用于在训练和验证时加载数据
train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)  # 训练数据加载器，启用数据打乱
val_loader = DataLoader(val_dataset, batch_size=32, shuffle=False)  # 验证数据加载器，无需打乱数据

# 实例化模型、定义损失函数和优化器
model = Net().to(device)
# 确保此路径指向你保存的模型状态文件
# model_path = 'COMP390Test1'
# model.load_state_dict(torch.load(model_path, map_location=device))
criterion = nn.CrossEntropyLoss()  # 使用交叉熵损失函数，适用于多分类问题
optimizer = optim.Adam(model.parameters(), lr=0.00001)  # 使用Adam优化器


num_epochs = 20  # 定义训练轮数
train_model(num_epochs, train_loader, val_loader, model, criterion, optimizer,model_path='no data enhancement )')  # 调用训练函数】





# 加载测试集
test_data_path = 'test_data'
test_dataset = datasets.ImageFolder(root='test_data', transform=data_transforms)
test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False)

# 开始测试
start_time = time.perf_counter()
test_model(test_loader, model, device)  
# test_model(test_loader, model)  # 使用测试数据和训练好的模型进行测试
end_time = time.perf_counter()
duration = end_time - start_time

print(f"The operation took {duration} seconds.")
