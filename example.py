import numpy as np  
  
# 读取数据  
data = np.load('frame_000.npy')  
  
# 将数据分为特征和标签  
X = data[:, :-1] # 假设最后一列是标签  
y = data[:, -1] # 假设最后一列是标签  
  
# 打印前5个样本  
print("前5个样本的特征：")  
print(X[:5])  
print("前5个样本的标签：")  
print(y[:5])