import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from sklearn.datasets import load_digits
from sklearn.model_selection import train_test_split
import numpy as np
import time

from torch.utils.tensorboard import SummaryWriter
# 创建SummaryWriter对象，指定日志保存目录​
writer = SummaryWriter('./my_experiment')

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
criterion = nn.NLLLoss()  # 负对数似然损失
optimizer = optim.Adam(model.parameters(), lr=0.05)
#optimizer = optim.SGD(model.parameters(), lr=0.001)
#criterion = CustomLoss(lambda_1=2.0, lambda_9=-0.5)  # 调整权重参数
# from torch.optim.lr_scheduler import StepLR
# scheduler = StepLR(optimizer, step_size=1, gamma=0.1)  # 每5个epoch，LR乘以0.1



# 训练模型
def train(epoch):
    model.train()
   
    start_time = time.time()
    running_loss = 0.0
    correct = 0
    total = 0
    
    for batch_idx, (data, target) in enumerate(train_loader):
        data, target = data.to(device), target.to(device)
        optimizer.zero_grad()
        output = model(data)
        loss = criterion(output, target)
        loss.backward()
        optimizer.step()
        
        running_loss += loss.item()
        
        # 计算准确率
        _, predicted = output.max(1)
        total += target.size(0)
        correct += predicted.eq(target).sum().item()
    
    end_time = time.time()
    avg_loss = running_loss / len(train_loader)
    accuracy = 100. * correct / total
    
    # 记录训练损失和准确率到TensorBoard
    # writer.add_scalar('Training Loss', avg_loss, epoch)
    # writer.add_scalar('Training Accuracy', accuracy, epoch)
    writer.add_scalars(
    'Loss Comparison',  # 图表标题
    {
        'Train': accuracy,
    },
    epoch
    )   

    print(f'Epoch {epoch+1}/{epochs}, 损失: {avg_loss:.4f}, 准确率: {accuracy:.2f}%, 耗时: {end_time-start_time:.2f}秒')
# 评估模型

    return accuracy




def test(epoch):
    model.eval()
    test_loss = 0
    correct = 0
    with torch.no_grad():
        for data, target in test_loader:
            data, target = data.to(device), target.to(device)
            output = model(data)
            test_loss += criterion(output, target).item()
            pred = output.argmax(dim=1, keepdim=True)  # 获取最大概率的索引
            correct += pred.eq(target.view_as(pred)).sum().item()

    test_loss /= len(test_loader)
    accuracy = 100. * correct / len(test_loader.dataset)
    print(f'测试集: 平均损失: {test_loss:.4f}, 准确率: {correct}/{len(test_loader.dataset)} ({accuracy:.2f}%)')
    writer.add_scalars(
    'Loss Comparison',  # 图表标题
    {
        'Test': accuracy,
    },
    epoch
    )   
# 训练和评估
print("开始训练CNN模型...")
epochs = 50 
for epoch in range(0,epochs):
    acc = train(epoch)
    #scheduler.step()  # 更新学习率
    test(epoch)


# 课堂作业， 输出测试集在每个 epoch 的 acc 和 loss