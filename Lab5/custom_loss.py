import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from sklearn.datasets import load_digits
from sklearn.model_selection import train_test_split
import numpy as np
import time

# 设置设备
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"使用设备: {device}")

# 加载sklearn中的手写数字数据集
digits = load_digits()
X = digits.data  # 特征数据 (1797, 64)
y = digits.target  # 标签数据 (1797,)

# 数据预处理：重塑为图像格式并归一化
X = X.reshape(-1, 1, 8, 8)  # 重塑为 (样本数, 通道数, 高度, 宽度)
X = X / 16.0  # 将像素值归一化到 [0,1] 范围

# 分割数据集
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# 转换为PyTorch张量
X_train = torch.FloatTensor(X_train)
y_train = torch.LongTensor(y_train)
X_test = torch.FloatTensor(X_test)
y_test = torch.LongTensor(y_test)

# 创建数据加载器
train_dataset = TensorDataset(X_train, y_train)
test_dataset = TensorDataset(X_test, y_test)
train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=1000)

# 定义CNN模型（适应8x8图像）
class CNN(nn.Module):
    def __init__(self):
        super(CNN, self).__init__()
        self.conv1 = nn.Conv2d(1, 10, kernel_size=3)  # 输入通道1，输出通道10
        self.conv2 = nn.Conv2d(10, 20, kernel_size=3)  # 输入通道10，输出通道20
        self.conv2_drop = nn.Dropout2d()  # 卷积层的Dropout
        self.fc1 = nn.Linear(20 * 1 * 1, 50)  # 全连接层
        self.fc2 = nn.Linear(50, 10)  # 输出层

    def forward(self, x):
        x = self.conv1(x)  # 卷积操作 (8-3+1=6) -> 6x6
        x = nn.functional.max_pool2d(x, 2)  # 最大池化 -> 3x3
        x = nn.functional.relu(x)  # ReLU激活函数
        
        x = self.conv2(x)  # 第二次卷积 (3-3+1=1) -> 1x1
        x = self.conv2_drop(x)  # Dropout防止过拟合
        x = nn.functional.max_pool2d(x, 1)  # 池化 (保持1x1)
        x = nn.functional.relu(x)  # ReLU激活函数
        
        x = x.view(-1, 20 * 1 * 1)  # 展平为一维向量
        x = self.fc1(x)  # 全连接层
        x = nn.functional.relu(x)  # ReLU激活函数
        x = nn.functional.dropout(x, training=self.training)  # Dropout
        x = self.fc2(x)  # 输出层
        return nn.functional.log_softmax(x, dim=1)  # 对数Softmax激活函数

# 初始化模型、损失函数和优化器
# 定义RNN模型
class RNN(nn.Module):
    def __init__(self, input_size=8, hidden_size=128, num_layers=2, num_classes=10):
        super(RNN, self).__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        
        # LSTM层
        self.rnn = nn.RNN(input_size, hidden_size, num_layers, batch_first=True)
        # # LSTM层
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True)
        
        # 全连接层
        self.fc = nn.Linear(hidden_size, num_classes)
        
    def forward(self, x):
        # 输入形状: (batch_size, 1, 8, 8)
        # 重塑为序列形式: (batch_size, 8, 8)
        batch_size = x.size(0)
        x = x.squeeze(1)  # 移除通道维度
        #print("x shape is ",x.shape)
        
        # 初始化隐藏状态
        h0 = torch.zeros(self.num_layers, batch_size, self.hidden_size).to(x.device)
        c0 = torch.zeros(self.num_layers, batch_size, self.hidden_size).to(x.device)
        
        # LSTM前向传播
        out, _ = self.lstm(x, (h0, c0))
        #out = self.rnn(x,(h0,c0))
        
        # 只使用最后一个时间步的输出
        out = self.fc(out[:, -1, :])
        return nn.functional.log_softmax(out, dim=1)

# 初始化RNN模型
model = RNN().to(device)
# criterion = nn.NLLLoss()
# optimizer = optim.Adam(model.parameters(), lr=0.001)


#model = CNN().to(device)
#criterion = nn.NLLLoss()  # 负对数似然损失
optimizer = optim.Adam(model.parameters(), lr=0.001)
class CustomLoss(nn.Module):
    def __init__(self, lambda_1=1.5, lambda_9=-0.5):
        super(CustomLoss, self).__init__()
        self.lambda_1 = lambda_1  # 增加数字1的权重
        self.lambda_9 = lambda_9  # 降低数字9的权重
        self.ce_loss = nn.CrossEntropyLoss(reduction='none')
        
    def forward(self, outputs, targets):
        # 标准交叉熵损失
        ce = self.ce_loss(outputs, targets)
        
        # 为数字1和9添加特殊权重
        batch_size = targets.size(0)
        for i in range(batch_size):
            if targets[i] == 1:
                ce[i] *= self.lambda_1  # 增加数字1的损失权重
            elif targets[i] == 9:
                ce[i] *= self.lambda_9  # 降低数字9的损失权重
                
        return ce.mean()
criterion = CustomLoss(lambda_1=2.0, lambda_9=0)  # 调整权重参数

# 训练模型
def train(epochs):
    model.train()
    for epoch in range(epochs):
        start_time = time.time()
        running_loss = 0.0
        for batch_idx, (data, target) in enumerate(train_loader):
            data, target = data.to(device), target.to(device)
            #print(data.shape) # [64,1,8,8]
            optimizer.zero_grad()
            output = model(data)
            loss = criterion(output, target)
            loss.backward()
            optimizer.step()
            
            running_loss += loss.item()
        
        end_time = time.time()
        print(f'Epoch {epoch+1}/{epochs}, 损失: {running_loss/len(train_loader):.4f}, 耗时: {end_time-start_time:.2f}秒')

# 评估模型
def test():
    model.eval()
    test_loss = 0
    correct = 0
    class_correct = [0] * 10  # 每个类别的正确预测数
    class_total = [0] * 10    # 每个类别的总样本数
    with torch.no_grad():
        for data, target in test_loader:
            data, target = data.to(device), target.to(device)
            output = model(data)
            test_loss += criterion(output, target).item()
            pred = output.argmax(dim=1, keepdim=True)  # 获取最大概率的索引
            correct += pred.eq(target.view_as(pred)).sum().item()
            # 计算每个类别的正确预测数
            for i in range(len(target)):
                label = target[i].item()
                class_correct[label] += pred[i].item() == label
                class_total[label] += 1
                if label == 1 or  pred[i].item()==1:
                    print("pred is ",pred[i].item()," label is ",label)

    test_loss /= len(test_loader)
    accuracy = 100. * correct / len(test_loader.dataset)
    # 打印每个类别的准确率
    for i in range(10):
        if class_total[i] > 0:
            print(f'Accuracy of class {i}: {100. * class_correct[i] / class_total[i]:.2f}% ({class_correct[i]}/{class_total[i]})')
        else:
            print(f'Accuracy of class {i}: N/A')
    
    # 特别关注类别1和9的准确率
    print(f'\nAccuracy of class 1: {100. * class_correct[1] / class_total[1]:.2f}%')
    print(f'Accuracy of class 9: {100. * class_correct[9] / class_total[9]:.2f}%')
    print(f'测试集: 平均损失: {test_loss:.4f}, 准确率: {correct}/{len(test_loader.dataset)} ({accuracy:.2f}%)')

# 训练和评估
print("开始训练CNN模型...")
train(epochs=20)  # 由于数据集较小，增加训练轮次
print("\n开始评估模型...")
test()    