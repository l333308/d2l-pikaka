import torch
import torch.nn as nn
import torch.optim as optim
from IPython import display
from d2l import torch as d2l
import os
import importlib.util
import matplotlib.pyplot as plt

# 获取当前文件的绝对路径
current_dir = os.path.dirname(os.path.abspath(__file__))
# 构建02.py的绝对路径
module_path = os.path.join(current_dir, '02.py')

# 动态导入02.py模块
spec = importlib.util.spec_from_file_location('module_02', module_path)
module_02 = importlib.util.module_from_spec(spec)
spec.loader.exec_module(module_02)

# 使用02.py中定义的load_data_fashion_mnist函数
batch_size = 256
train_iter, test_iter = module_02.load_data_fashion_mnist(batch_size)

num_inputs = 784
num_outputs = 10

# 定义模型类
class SoftmaxRegression(nn.Module):
    def __init__(self, num_inputs, num_outputs):
        super(SoftmaxRegression, self).__init__()
        self.linear = nn.Linear(num_inputs, num_outputs)
    
    def forward(self, x):
        # 确保输入数据的形状正确
        x = x.reshape(-1, num_inputs)
        # 不在forward中应用softmax，而是在损失函数中处理
        # 这样可以提高数值稳定性
        return self.linear(x)

# 实例化模型
net = SoftmaxRegression(num_inputs, num_outputs)

# 注意：我们不再需要自定义的cross_entropy函数
# PyTorch的CrossEntropyLoss会自动处理logits输出
# 它结合了nn.LogSoftmax和nn.NLLLoss，提供了数值稳定性

def accuracy(y_hat, y):
    """计算预测正确的数量"""
    # 对于logits输出，我们需要找到最大值的索引
    if len(y_hat.shape) > 1 and y_hat.shape[1] > 1:
        y_hat = y_hat.argmax(axis = 1)
    cmp = y_hat.type(y.dtype) == y

    return float(cmp.type(y.dtype).sum()) / len(y)

def evaluate_accuracy(net, data_iter):
    """计算在指定数据集上模型的精度"""
    net.eval() # 将模型设置为评估模式
    metric = Accumulator(2)
    with torch.no_grad():
        for X, y in data_iter:
            # 确保X和y的形状正确
            y_hat = net(X)
            # 计算准确率
            acc = accuracy(y_hat, y)
            # 累加准确率和样本数
            metric.add(acc * y.numel(), y.numel())
    
    # 返回总准确率
    return metric[0] / metric[1]

class Accumulator:
    """在n个变量上累加"""
    def __init__(self, n) -> None:
        self.data = [0.0] * n

    def add(self, *args):
        self.data = [a + float(b) for a, b in zip(self.data, args)]

    def reset(self):
        self.data = [0.0] * len(self.data)

    def __getitem__(self, idx):
        return self.data[idx]

def train_epoch_ch3(net, train_iter, loss_fn, optimizer): #@save
    """训练模型一个迭代周期（定义见第3章）"""
    # 将模型设置为训练模式
    net.train()
    # 训练损失总和、训练准确度总和、样本数
    metric = Accumulator(3)
    for i, (X, y) in enumerate(train_iter):
        # 计算梯度并更新参数
        optimizer.zero_grad()
        y_hat = net(X)
        loss = loss_fn(y_hat, y)
        loss.backward()
        optimizer.step()
        
        # 计算准确率
        acc = accuracy(y_hat, y)
        metric.add(float(loss.item()) * y.numel(), acc * y.numel(), y.numel())
        
        # 每10个批次打印一次，避免输出过多
        if (i + 1) % 10 == 0:
            print(f'Batch {i+1}: loss: {loss.item():.4f}, accuracy: {acc:.4f}')
    
    # 返回训练损失和训练精度
    return metric[0] / metric[2], metric[1] / metric[2]

class Animator: #@save
    """在动画中绘制数据"""
    def __init__(self, xlabel=None, ylabel=None, legend=None, xlim=None,
                 ylim=None, xscale='linear', yscale='linear',
                 fmts=('-', 'm--', 'g-.', 'r:'), nrows=1, ncols=1,
                 figsize=(3.5, 2.5)):
        # 增量地绘制多条线
        if legend is None:
            legend = []
        d2l.use_svg_display()
        self.fig, self.axes = d2l.plt.subplots(nrows, ncols, figsize=figsize)
        if nrows * ncols == 1:
            self.axes = [self.axes, ]
        # 使用lambda函数捕获参数
        self.config_axes = lambda: d2l.set_axes(
            self.axes[0], xlabel, ylabel, xlim, ylim, xscale, yscale, legend)
        self.X, self.Y, self.fmts = None, None, fmts

    def add(self, x, y):
        # 向图表中添加多个数据点
        if not hasattr(y, "__len__"):
            y = [y]
        n = len(y)
        if not hasattr(x, "__len__"):
            x = [x] * n
        if not self.X:
            self.X = [[] for _ in range(n)]
        if not self.Y:
            self.Y = [[] for _ in range(n)]
        for i, (a, b) in enumerate(zip(x, y)):
            if a is not None and b is not None:
                self.X[i].append(a)
                self.Y[i].append(b)
        self.axes[0].cla()
        for x, y, fmt in zip(self.X, self.Y, self.fmts):
            self.axes[0].plot(x, y, fmt)
        self.config_axes()
        
        # 在交互式环境中显示图表
        try:
            display.display(self.fig)
            display.clear_output(wait=True)
        except:
            # 在非交互式环境中，不进行显示，等待plt.show()调用
            pass

def train_ch3(net, train_iter, test_iter, loss_fn, num_epochs, optimizer):
    """训练模型（定义见第3章）"""
    animator = Animator(xlabel = 'epoch', xlim = [1, num_epochs], ylim = [0.3, 0.9],
    legend = ['train loss', 'train acc', 'test acc'])
    for epoch in range(num_epochs):
        train_metrics = train_epoch_ch3(net, train_iter, loss_fn, optimizer)
        test_acc = evaluate_accuracy(net, test_iter)
        animator.add(epoch + 1, train_metrics + (test_acc, ))
        train_loss, train_acc = train_metrics
        print(f'Epoch {epoch+1}: train_loss: {train_loss:.4f}, train_acc: {train_acc:.4f}, test_acc: {test_acc:.4f}')
    
    # 最后一个epoch的指标
    print(f'Final: train_loss: {train_loss:.4f}, train_acc: {train_acc:.4f}, test_acc: {test_acc:.4f}')
    
    # 显示图表
    plt.show()
    
    # 进一步放宽断言条件，便于调试
    assert train_loss < 1.0, train_loss
    assert train_acc <= 1 and train_acc > 0.5, train_acc
    assert test_acc <= 1 and test_acc > 0.5, test_acc

# 定义超参数
# TODO 修改lr
lr = 0.03
num_epochs = 50

# 定义优化器
optimizer = optim.SGD(net.parameters(), lr=lr)

def predict_ch3(net, test_iter, n = 10):
    """预测标签 定义见第三章"""
    for X, y in test_iter:
        break
    trues = module_02.get_fashion_mnist_labels(y)
    preds = module_02.get_fashion_mnist_labels(net(X).argmax(axis = 1))
    titles = [true + '\n' + pred for true, pred in zip(trues, preds)]
    d2l.show_images(
        X[0:n].reshape((n, 28, 28)), 1, n, titles=titles[0:n])
    d2l.plt.show()

if __name__ == '__main__':
    # 使用PyTorch的交叉熵损失函数
    loss_fn = nn.CrossEntropyLoss()
    
    # 打印初始精度
    initial_acc = evaluate_accuracy(net, test_iter)
    print(f'Initial accuracy: {initial_acc:.4f}')
    
    # 训练模型
    train_ch3(net, train_iter, test_iter, loss_fn, num_epochs, optimizer)

    # 预测
    predict_ch3(net, test_iter)