import torch
import torch.nn as nn
import matplotlib.pyplot as plt

# 定义模型各层、批量大小
n_in, n_h, n_out, batch_size = 10, 5, 1, 10

# 创建虚拟输入数据和目标数据
x = torch.randn(batch_size, n_in)
y = torch.tensor([[1.0], [0.0], [0.0], [1.0], [1.0], [1.0], [0.0], [0.0], [1.0], [1.0]])

# 创建顺序模型，包含线性层、ReLU激活函数和Sigmoid激活函数
model = nn.Sequential(
    nn.Linear(n_in, n_h),   # 输入层到隐藏层的线性变换
    nn.ReLU(),  # 隐藏层的ReLU激活函数
    nn.Linear(n_h, n_out),  # 隐藏层到输出层的线性变换
    nn.Sigmoid()    # 输出层的Sigmoid激活函数
)

# 均方误差损失函数 随机梯度下降优化器
criterion = nn.MSELoss()
optimizer = torch.optim.SGD(model.parameters(), lr = 0.01)

# 用于存储每轮的损失
losses = []

# 梯度下降法 训练模型
for epoch in range(1000):
    y_pred = model(x) # 前向传播 计算预测值
    loss = criterion(y_pred, y) # 计算损失
    losses.append(loss.item()) # 记录损失
    print(f'Epoch [{epoch + 1 / 1000:.0f}], Loss: {loss.item():.4f}')

    optimizer.zero_grad() # 清零梯度
    loss.backward() # 反向传播，计算梯度
    optimizer.step() # 更新参数

# 可视化损失曲线
plt.figure(figsize = (8, 5))
plt.plot(range(1, 1001), losses, label = 'Loss')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.title('Loss over epochs')
plt.legend()
plt.grid()
plt.show()
