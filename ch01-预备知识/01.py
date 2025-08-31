import torch
import os
import pandas as pd
import d2l

x = torch.tensor([1.0, 2, 4, 8])
y = torch.tensor([2, 2, 2, 2])
print(x + y, x - y, x * y, x / y, x ** y)

X = torch.arange(12, dtype=torch.float32).reshape((3,4))
Y = torch.tensor([[2.0, 1, 4, 3], [1, 2, 3, 4], [4, 3, 2, 1]])
print("\n")
print(X)
print(Y)
print("\n")
print(torch.cat((X, Y), dim=0))
print(torch.cat((X, Y), dim=1))
print(X < Y)
print(X.sum())

A = X.numpy()
B = torch.tensor(A)
print("\n")
print(A, type(A))
print(B, type(B))

os.makedirs(os.path.join('..', 'data'), exist_ok=True)
data_file = os.path.join('..', 'data', 'house_tiny.csv')
with open(data_file, 'w') as f:
    f.write('NumRooms,Alley,Price\n') # 列名
    f.write('NA,Pave,127500\n') # 每行表示一个数据样本
    f.write('2,NA,106000\n')
    f.write('4,NA,178100\n')
    f.write('NA,NA,140000\n')
data = pd.read_csv(data_file)
print("\n")
print(data)
inputs, outputs = data.iloc[:, 0:2], data.iloc[:, 2]
# 统计每列的缺失值数量
na_count = inputs.isna().sum()
print("每列的缺失值数量：")
print(na_count)

# 找出缺失值最多的列
most_missing_col = na_count.idxmax()
print(f"\n缺失值最多的列是: {most_missing_col}, 缺失值数量: {na_count[most_missing_col]}")

# 删除缺失值最多的列
inputs_dropped = inputs.drop(columns=[most_missing_col])
print("\n删除缺失值最多的列后的数据：")
print(inputs_dropped)

# 对剩余列进行缺失值填充
inputs_clean = inputs_dropped.copy()
for column in inputs_clean.columns:
    if inputs_clean[column].dtype == 'object':
        # 对于字符串类型的列，用最常见的值填充
        inputs_clean[column] = inputs_clean[column].fillna(inputs_clean[column].mode()[0] if not inputs_clean[column].mode().empty else 'NA')
    else:
        # 对于数值类型的列，用平均值填充
        inputs_clean[column] = inputs_clean[column].fillna(inputs_clean[column].mean())

print("\n填充缺失值后的数据：")
print(inputs_clean)
# 对处理后的数据进行独热编码
inputs_encoded = pd.get_dummies(inputs_clean, dummy_na=True)
print("\n独热编码后的数据：")
print(inputs_encoded)

print("\n")
t = torch.arange(20).reshape(5, 4)
print(t)
print(t.T)