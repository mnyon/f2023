import numpy as np
import tensorflow as tf
from tensorflow.keras import layers, models

# 建立深度学习模型
def build_model(input_shape):
    model = models.Sequential()
    # 添加卷积层
    model.add(layers.Conv2D(64, (3, 3), activation='relu', input_shape=input_shape))
    model.add(layers.MaxPooling2D((2, 2)))
    model.add(layers.Conv2D(128, (3, 3), activation='relu'))
    model.add(layers.MaxPooling2D((2, 2)))
    model.add(layers.Conv2D(128, (3, 3), activation='relu'))
    # 添加全连接层
    model.add(layers.Flatten())
    model.add(layers.Dense(128, activation='relu'))
    model.add(layers.Dense(1, activation='linear'))  # 输出层，用于预测降水量

    # 编译模型
    model.compile(optimizer='adam', loss='mean_squared_error', metrics=['mae'])
    
    return model

# 准备训练数据和标签
# X_train是输入数据，包括ZH、ZDR等观测变量，形状为 (样本数, 时间步长, 高度, 宽度, 通道数)
# y_train是输出数据，即降水量，形状为 (样本数,)

# 创建模型
input_shape = (10, height, width, num_channels)  # 根据实际数据维度调整
model = build_model(input_shape)

# 训练模型
model.fit(X_train, y_train, epochs=10, batch_size=32, validation_split=0.2)

# 使用模型进行预测
y_pred = model.predict(X_test)

# 评估模型性能，例如计算均方根误差(RMSE)等

