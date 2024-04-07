# 导入必要的库
import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms
from torch.utils.data import DataLoader, random_split
import torch.nn.functional as F
import time
import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import confusion_matrix, classification_report
import matplotlib.pyplot as plt
import seaborn as sns  # 确保你的环境中安装了seaborn

# 这里得bug有可能导致环境崩溃！！！！并不是一个好的解决方法
import os
os.environ['KMP_DUPLICATE_LIB_OK']='TRUE'

# 环境配置和全局变量
dataset_path = 'train_data_3'
test_data_path = 'test_data_3'
# dataset_path = 'ID_train'
# test_data_path = 'ID_test'
torch.manual_seed(0)  # 确保可复现性
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")  # 检查并设置设备
print(f"Using {device} device")

# 数据预处理
def get_transforms():
    return transforms.Compose([
        # transforms.Resize((30, 150)), # 将图像大小统一调整为150x150
        # 重置ID的resize
        transforms.Resize((30, 150)),
        transforms.Grayscale(num_output_channels=1),
        # transforms.RandomRotation(20), # 随机旋转图像，角度在-20到20度之间    数据增强1
        # transforms.RandomHorizontalFlip(),  # 随机进行水平翻转，以增加数据多样性
        # transforms.RandomAffine(degrees=0, scale=(0.8, 1.2)), # 数据增强2
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485], std=[0.229]),
    ])

# 加载数据集
def load_datasets(dataset_path, transform):
    CUSTOM_CLASS_TO_IDX = {'A': 0, 'B': 1, 'C': 2, 'D':3, 'E':4, 'None':5}
    # CUSTOM_CLASS_TO_IDX = {'1_file': 1, '2_file': 2, '3_file': 3, '4_file':4, '5_file':5, '6_file':6, '7_file':7, '8_file':8, '9_file':9}
    full_dataset = datasets.ImageFolder(root=dataset_path, transform=transform)
    full_dataset.class_to_idx = CUSTOM_CLASS_TO_IDX
    return full_dataset

# 数据加载器
def get_dataloaders(train_dataset,batch_size=8):
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    return train_loader

# 定义问题的神经网络模型
class Questions_model(nn.Module):
    def __init__(self):
        super(Questions_model, self).__init__()
        # 定义卷积层和全连接层
        self.conv1 = nn.Conv2d(1, 32, kernel_size=3, padding=1)  # 第一个卷积层
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

# 训练和验证函数
def train_model(model, num_epochs, train_loader, criterion, optimizer, scheduler, device, model_path):# 对模型进行命名！！！！！
    # 初始化存储每个epoch的训练损失和准确率的列表
    train_losses = []
    train_accuracies = []

    # 开始训练循环
    for epoch in range(num_epochs):
        model.train()  # 将模型设置为训练模式
        running_loss = 0.0  # 累积当前epoch的总损失
        correct = 0  # 累积当前epoch中正确预测的样本数
        total = 0  # 累积当前epoch处理的总样本数

        # 迭代当前epoch的训练数据
        for images, labels in train_loader:
            images, labels = images.to(device), labels.to(device)

            optimizer.zero_grad()  # 清零梯度
            outputs = model(images)  # 前向传播：计算模型输出
            loss = criterion(outputs, labels)  # 计算损失
            loss.backward()  # 反向传播：计算梯度
            optimizer.step()  # 优化步骤：更新权重

            running_loss += loss.item() * images.size(0)  # 更新总损失
            _, predicted = torch.max(outputs.data, 1)  # 获取预测结果
            total += labels.size(0)  # 更新总样本数
            correct += (predicted == labels).sum().item()  # 更新正确预测的样本数

        # 计算当前epoch的平均损失和准确率
        epoch_loss = running_loss / total
        epoch_acc = correct / total

        # 记录当前epoch的损失和准确率
        train_losses.append(epoch_loss)
        train_accuracies.append(epoch_acc)

        print(f'Epoch {epoch + 1}/{num_epochs}, Train Loss: {epoch_loss:.4f}, Train Accuracy: {epoch_acc:.4f}')
        scheduler.step()  # 更新学习率

    # 保存模型参数到指定的文件
    torch.save(model.state_dict(), model_path)
    print(f'Model saved to {model_path}')
    # plot_metrics(train_losses, train_accuracies, 'training_metrics.png')

def load_test_data(test_data_path, transform):
    test_dataset = datasets.ImageFolder(root=test_data_path, transform=transform)
    test_loader = DataLoader(test_dataset, batch_size=8, shuffle=False)
    return test_loader

def test_model(test_loader, model, device):
    model.eval()  # 设置模型为评估模式
    correct = 0
    total = 0
    misclassified = []  # 用于记录误分类的信息
    all_labels = []  # 收集所有真实标签
    all_preds = []  # 收集所有预测标签

    with torch.no_grad():  # 不计算梯度，加速推理
        for batch_idx, (images, labels) in enumerate(test_loader):
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            _, predicted = torch.max(outputs, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

            # 收集标签和预测结果
            all_labels.extend(labels.cpu().numpy())
            all_preds.extend(predicted.cpu().numpy())

            # 检查哪些图片分类错误了
            mismatches = (predicted != labels)
            mis_idxs = np.where(mismatches.cpu().numpy())[0]
            for idx in mis_idxs:
                img_path = test_loader.dataset.samples[batch_idx * test_loader.batch_size + idx][0]
                img_name = os.path.basename(img_path)
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

    # 返回包括所有真实标签和预测标签的列表
    return test_accuracy, misclassified, all_labels, all_preds

def visualize_performance(y_true, y_pred, class_names):
    cm = confusion_matrix(y_true, y_pred)
    cr = classification_report(y_true, y_pred, target_names=class_names)
    print("Classification Report:\n", cr)

    fig, ax = plt.subplots(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', ax=ax, cmap='Blues', 
                xticklabels=class_names, yticklabels=class_names)
    plt.ylabel('Actual')
    plt.xlabel('Predicted')
    plt.title('Confusion Matrix')
    plt.show()


# 主执行逻辑
def main_train(model_filename):
    transforms = get_transforms()
    train_dataset= load_datasets(dataset_path, transforms)
    train_loader= get_dataloaders(train_dataset)
    model = Questions_model().to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.0005)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=10, gamma=0.1)
    num_epochs = 45
    
    # 训练数据
    train_model(model, num_epochs, train_loader, criterion, optimizer, scheduler, device, model_path=model_filename)

    # 测试模型
    test_loader = load_test_data(test_data_path, transforms)
    model.load_state_dict(torch.load(model_filename))  # 确保已加载模型参数
    model.to(device)
    test_accuracy, misclassified, y_true, y_pred = test_model(test_loader, model, device)

    # 计算和可视化性能指标...
    class_names = test_loader.dataset.classes
    visualize_performance(y_true, y_pred, class_names)



if __name__ == "__main__":
    model_filename = 'bz8_lr0.0005_ep45_3'  # 模型文件名
    # model_filename = 'ID_lr0.0005_ep10'
    main_train(model_filename)
    # main_notrain(model_filename)




def main_notrain(model_filename):
    # 设置变换、加载测试集等
    transforms = get_transforms()
    test_dataset = datasets.ImageFolder(root=test_data_path, transform=transforms)
    test_loader = DataLoader(test_dataset, batch_size=8, shuffle=False)

    # 初始化模型并加载预训练的权重
    model = Questions_model().to(device)
    model.load_state_dict(torch.load(model_filename))  # 确保替换为你的模型文件路径
    model.to(device)

    # 测试模型
    test_model(test_loader, model, device)


# 绘制训练和验证的损失和准确率
def plot_metrics(train_loss, train_acc, save_path):
    epochs = range(1, len(train_loss) + 1)
    plt.figure(figsize=(12, 4))
    plt.subplot(1, 2, 1)
    plt.plot(epochs, train_loss, 'r-', label='Training Loss')
    plt.title('Training Loss')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.legend()
    plt.subplot(1, 2, 2)
    plt.plot(epochs, train_acc, 'r-', label='Training Accuracy')
    plt.title('Training Accuracy')
    plt.xlabel('Epochs')
    plt.ylabel('Accuracy')
    plt.legend()
    plt.savefig(save_path)
    plt.show()