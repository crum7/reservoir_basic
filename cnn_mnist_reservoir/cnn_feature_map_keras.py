import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers



'''
論文のモデル実装のCNN部分
Reservoir Computing with Untrained Convolutional Neural Networks for Image Recognition
https://ieeexplore.ieee.org/stamp/stamp.jsp?tp=&arnumber=8545471&tag=1

MNISTのデータセットを読み込み、特徴マップを作り保存する
'''
# MNISTデータの読み込み
(x_train, y_train), (x_test, y_test) = keras.datasets.mnist.load_data()

# ピクセル値を0から1の範囲に正規化
x_train = x_train.astype("float32") / 255
x_test = x_test.astype("float32") / 255

# 特徴抽出部分のCNNモデルを定義
model = keras.Sequential(
    [
        layers.InputLayer(input_shape=(28, 28, 1)),
        layers.Conv2D(
            84,
            kernel_size=(3, 3),
            strides=(1, 1),
            kernel_initializer=tf.keras.initializers.RandomUniform(minval=-0.06, maxval=0.06),
            bias_initializer='zeros',
            activation='tanh'
        ),
        layers.MaxPooling2D(pool_size=(3, 3), strides=(2, 2)),
        layers.Conv2D(
            84,
            kernel_size=(3, 3),
            strides=(1, 1),
            kernel_initializer=tf.keras.initializers.RandomUniform(minval=-0.06, maxval=0.06),
            bias_initializer='zeros',
            activation='tanh'
        ),
        layers.AveragePooling2D(pool_size=(3, 3), strides=(2, 2)),
        layers.Conv2D(
            84,
            kernel_size=(3, 3),
            strides=(1, 1),
            kernel_initializer=tf.keras.initializers.RandomUniform(minval=-0.06, maxval=0.06),
            bias_initializer='zeros',
            activation='tanh'
        ),
    ]
)

# テストデータの特徴マップを抽出
feature_maps_test = model.predict(x_test.reshape(-1, 28, 28, 1))
feature_maps_train = model.predict(x_train.reshape(-1, 28, 28, 1))

# 特徴マップのサイズを確認
print("Feature maps shape:", feature_maps_test.shape)
print("Feature maps shape:", feature_maps_train.shape)


# 特徴マップとラベルを保存
np.save('feature_maps_test.npy', feature_maps_test)
np.save('feature_maps_train.npy', feature_maps_train)

print("y_test shape:", y_test.shape)
print("y_train shape:", y_train.shape)

np.save('labels_test.npy', y_test)
np.save('labels_train.npy', y_train)

