# 导入必要的库
import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms
from torch.utils.data import DataLoader, random_split
import torch.nn.functional as F
import time
# 设定数据集路径（这需要根据实际情况进行替换）
dataset_path = 'Questions'
# 确保随机划分数据集的可复现性
torch.manual_seed(0)  # 设置随机种子


# 图像预处理和增强 - 定义了一个转换流程，用于对图像数据进行处理和增强
data_transforms = transforms.Compose([
    transforms.Resize((30, 150)),  # 将图像大小统一调整为150x150
    # transforms.RandomHorizontalFlip(),  # 随机进行水平翻转，以增加数据多样性
    # transforms.RandomRotation(20),  # 随机旋转图像，角度在-20到20度之间
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

# 实例化模型、定义损失函数和优化器
model = Net()
criterion = nn.CrossEntropyLoss()  # 使用交叉熵损失函数，适用于多分类问题
optimizer = optim.Adam(model.parameters(), lr=0.001)  # 使用Adam优化器

# 训练和验证模型
def train_model(num_epochs, train_loader, val_loader, model, criterion, optimizer, model_path='COMP390Test1'):
    for epoch in range(num_epochs):  # 循环每一个epoch
        model.train()  # 将模型设置为训练模式
        running_loss = 0.0
        for images, labels in train_loader:  # 遍历训练数据
            optimizer.zero_grad()  # 清零梯度
            outputs = model(images)  # 前向传播
            loss = criterion(outputs, labels)  # 计算损失
            loss.backward()  # 反向传播
            optimizer.step()  # 更新参数

            running_loss += loss.item() * images.size(0)  # 累计损失
        epoch_loss = running_loss / len(train_loader.dataset)  # 计算平均损失

        # 在验证集上评估模型
        model.eval()  # 将模型设置为评估模式
        correct = 0
        total = 0
        with torch.no_grad():  # 不计算梯度，以加速和减少内存使用
            for images, labels in val_loader:  # 遍历验证数据
                outputs = model(images)
                _, predicted = torch.max(outputs.data, 1)  # 获取预测结果
                total += labels.size(0)
                correct += (predicted == labels).sum().item()  # 计算正确预测的数量
            val_accuracy = correct / total  # 计算验证集上的准确率

        print(f'Epoch {epoch + 1}/{num_epochs}, Loss: {epoch_loss:.4f}, Validation Accuracy: {val_accuracy:.4f}')
    # 保存模型参数
    torch.save(model.state_dict(), model_path)
    print(f'Model saved to {model_path}')

# 加载测试集
test_data_path = 'test_data'
test_dataset = datasets.ImageFolder(root='test_data', transform=data_transforms)
test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False)

# 测试模型的函数
def test_model(test_loader, model):
    model.eval()  # 设置模型为评估模式
    correct = 0
    total = 0
    with torch.no_grad():  # 不计算梯度，加速推理
        for images, labels in test_loader:
            outputs = model(images)
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
    test_accuracy = correct / total
    print(f'Test Accuracy: {test_accuracy:.4f}')

num_epochs = 10  # 定义训练轮数

train_model(num_epochs, train_loader, val_loader, model, criterion, optimizer,model_path='COMP390Test1')  # 调用训练函数】


start_time = time.perf_counter()
test_model(test_loader, model)  # 使用测试数据和训练好的模型进行测试
end_time = time.perf_counter()
duration = end_time - start_time

print(f"The operation took {duration} seconds.")