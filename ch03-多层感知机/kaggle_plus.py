import os
import pandas as pd
import numpy as np
import torch
from torch import nn
from d2l import torch as d2l

"""加载训练、测试数据集"""
curdir = os.path.dirname(os.path.abspath(__file__))
file_path = f'{curdir}/../../data/kaggle'
train_data = pd.read_csv(f'{file_path}/train.csv')
test_data = pd.read_csv(f'{file_path}/test.csv')
print(train_data.shape)
print(test_data.shape)
print(train_data.iloc[0:4].SalePrice)
print(train_data.iloc[0:4].SalePrice.values)

# 查看训练集、测试集 前4个、最后2个特征，以及相应标签（房价）
print(train_data.iloc[0:4, [0, 1, 2, 3, -3, -2, -1]])
print(test_data.iloc[0:4, [0, 1, 2, 3, -3, -2, -1]])

all_features = pd.concat((train_data.iloc[:, 1:-1], test_data.iloc[:, 1:]))
# 2919,79
print(f'all_features.shape: {all_features.shape}')

# 打印all_features的前4个样本的若干列的dtypes int64、object、float64、int64...
# dtypes不为object，即都为数值类型
#print(all_features.iloc[0:4, [0, 1, 2, 3, 4, -3, -2, -1]].dtypes)

# 数值类型的列 数据标准化
numeric_features = all_features.dtypes[all_features.dtypes != 'object'].index
all_features[numeric_features] = all_features[numeric_features].apply(
    lambda x: (x - x.mean()) / (x.std()))
# 标准化后，每个数值特征的均值变为0，所以可以直接用0来替换缺失值
all_features[numeric_features] = all_features[numeric_features].fillna(0)

# 使用get_dummies将分类变量转换为独热编码，dummy_na=True将缺失值也视为一个类别
# MSZoning列下 原有RL、RM两种值 转为one-hot编码 创建两个新的特征MSZoning_RL、MSZoning_RM
all_features = pd.get_dummies(all_features, dummy_na=True)
print(f'all_features.shape: {all_features.shape}')

# 确保所有数据都是数值类型
all_features = all_features.astype('float32')

# pandas格式中提取numpy格式，并转换为张量用于训练
n_train = train_data.shape[0]
train_features = torch.tensor(all_features[:n_train].values, dtype=torch.float32)
test_features = torch.tensor(all_features[n_train:].values, dtype=torch.float32)
train_labels = torch.tensor(train_data.SalePrice.values.reshape(-1, 1), dtype=torch.float32)

# 定义损失函数
loss = nn.MSELoss()
in_features = train_features.shape[1]

# 改进1：使用多层感知机替代单层线性网络
def get_net():
    # 定义一个三层的神经网络，包含两个隐藏层
    # 调整 Dropout 率到更合理的水平
    net = nn.Sequential(
        nn.Linear(in_features, 256),  # 第一个隐藏层，256个神经元
        nn.ReLU(),                    # ReLU激活函数
        nn.BatchNorm1d(256),          # 批量归一化
        nn.Dropout(0.3),             # 适度的 Dropout 率
        nn.Linear(256, 128),          # 第二个隐藏层，128个神经元
        nn.ReLU(),                    # ReLU激活函数
        nn.BatchNorm1d(128),          # 批量归一化
        nn.Dropout(0.2),             # 较低的 Dropout 率
        nn.Linear(128, 1)             # 输出层
    )
    return net

def log_rmse(net, features, labels):
    # 为了在取对数时进一步稳定该值，将小于1的值设置为1
    clipped_preds = torch.clamp(net(features), 1, float('inf'))
    rmse = torch.sqrt(loss(torch.log(clipped_preds), torch.log(labels)))

    return rmse.item()

def train(net, train_features, train_labels, test_features, test_labels,
    num_epochs, learning_rate, weight_decay, batch_size):
    train_ls, test_ls = [], []
    train_iter = d2l.load_array((train_features, train_labels), batch_size)

    # 改进2：使用AdamW优化器和学习率调度器
    optimizer = torch.optim.AdamW(net.parameters(), lr=learning_rate, weight_decay=weight_decay, betas=(0.9, 0.999))
    
    # 学习率调度器：在训练过程中逐渐降低学习率
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='min', factor=0.5, patience=10, verbose=True)
    
    for epoch in range(num_epochs):
        # 训练模式
        net.train()
        for X, y in train_iter:
            optimizer.zero_grad()
            l = loss(net(X), y)
            l.backward()
            # 梯度裁剪，防止梯度爆炸
            nn.utils.clip_grad_norm_(net.parameters(), max_norm=1.0)
            optimizer.step()
        
        # 评估模式
        net.eval()
        with torch.no_grad():
            train_rmse = log_rmse(net, train_features, train_labels)
            train_ls.append(train_rmse)
            if test_labels is not None:
                test_rmse = log_rmse(net, test_features, test_labels)
                test_ls.append(test_rmse)
                # 根据验证集损失调整学习率
                scheduler.step(test_rmse)

    return train_ls, test_ls

def get_k_fold_data(k, i, X, y):
    assert k > 1
    fold_size = X.shape[0] // k
    X_train, y_train = None, None
    for j in range(k):
        idx = slice(j * fold_size, (j + 1) * fold_size)
        X_part, y_part = X[idx, :], y[idx]
        if j == i:
            X_valid, y_valid = X_part, y_part
        elif X_train is None:
            X_train, y_train = X_part, y_part
        else:
            X_train = torch.cat((X_train, X_part), dim=0)
            y_train = torch.cat((y_train, y_part), dim=0)
    
    return X_train, y_train, X_valid, y_valid


def k_fold(k, X_train, y_train, num_epochs, learning_rate, weight_decay, batch_size):
    train_l_sum, valid_l_sum = 0, 0
    """x_train 训练集特征，y_train 训练集标签"""
    for i in range(k):
        data = get_k_fold_data(k, i, X_train, y_train)
        net = get_net()
        train_ls, valid_ls = train(net, *data, num_epochs, learning_rate, weight_decay, batch_size)
        train_l_sum += train_ls[-1]
        valid_l_sum += valid_ls[-1]
        if i == 0:
            d2l.plot(list(range(1, num_epochs + 1)), [train_ls, valid_ls], xlabel='epoch', ylabel='rmse',
            legend=['train', 'valid'], yscale='log')
        print(f'\n折{i + 1}，训练log rmse{float(train_ls[-1]):f}, '
            f'验证log rmse{float(valid_ls[-1]):f}')
    d2l.plt.show()
    
    return train_l_sum / k, valid_l_sum / k

# 改进2：调整超参数 - 回归有效配置
# 基于原始kaggle.py的有效参数进行微调
# 使用接近原始的学习率和正则化设置
k, num_epochs, lr, weight_decay, batch_size = 5, 200, 0.01, 0.001, 32
train_l, valid_l = k_fold(k, train_features, train_labels, num_epochs, lr, weight_decay, batch_size)

print(f'\n{k}-折验证: 平均训练log rmse: {float(train_l):f}, '
    f'平均验证log rmse: {float(valid_l):f}')
 
def train_and_pred(train_features, test_features, train_labels, test_data,
    num_epochs, lr, weight_decay, batch_size):
    net = get_net()
    train_ls, _ = train(net, train_features, train_labels, None, None,
                        num_epochs, lr, weight_decay, batch_size)
    d2l.plot(np.arange(1, num_epochs + 1), [train_ls], xlabel='epoch',
             ylabel='log rmse', yscale='log')
    d2l.plt.show()
    print(f'训练log rmse：{float(train_ls[-1]):f}')
    # 将网络应用于测试集。
    preds = net(test_features).detach().numpy()
    # 将其重新格式化以导出到Kaggle
    test_data['SalePrice'] = pd.Series(preds.reshape(1, -1)[0])
    submission = pd.concat([test_data['Id'], test_data['SalePrice']], axis=1)
    csv_path = f'{curdir}/submission_plus.csv'
    submission.to_csv(csv_path, index=False)

# 使用相同的超参数进行最终训练和预测
train_and_pred(train_features, test_features, train_labels, test_data, num_epochs, lr, weight_decay, batch_size)