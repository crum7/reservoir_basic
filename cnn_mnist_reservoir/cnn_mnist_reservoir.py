#特徴マップ作成時に使用
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
#ESNで使用
from sklearn.metrics import accuracy_score, f1_score, make_scorer
from sklearn.model_selection import RandomizedSearchCV
from scipy.stats import uniform
from pyrcn.echo_state_network import ESNClassifier
from pyrcn.model_selection import SequentialSearchCV


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


#特徴マップを利用して、ESNで学習

#(データ数,2*2*84)に変換する。
feature_maps_train = feature_maps_train.reshape(60000, -1)
feature_maps_test = feature_maps_test.reshape(10000, -1)


# 読み込んだデータの形状を表示
print('Feature maps shape:', feature_maps_train.shape)
print('Labels shape:', y_train.shape)

# データの分割
train_X=feature_maps_train
test_X = feature_maps_test
train_Y = y_train
test_Y = y_test

scorer = make_scorer(accuracy_score, greater_is_better=True)

step_1_params = {
    'input_scaling': uniform(0, 1),
    'hidden_layer_size': [340,350,360,370,380,390,400,410,420],
    'spectral_radius': uniform(0, 1.5),
    'leaking_rate': uniform(0, 1),
}

#n_iter:パラメータの組み合わせを選ぶ際の試行回数
#cv : Cross Validation（交差検証）の分割数

kwargs_1 = {
    'n_iter': 300, 'n_jobs': -1, 'scoring': scorer,'cv': 3
}

searches = [('step1', RandomizedSearchCV, step_1_params, kwargs_1)]

# ESNのハイパーパラメータを固定
esn = ESNClassifier(reservoir_activation='tanh', decision_strategy='vote')
esn_opti = SequentialSearchCV(esn, searches,error_score='raise').fit(
    train_X, train_Y)
print('esn_opti',esn_opti)
best_params = esn_opti.best_estimator_.get_params()
print(best_params)

y_pred_classes = esn_opti.predict(X=test_X)
y_pred_proba = esn_opti.predict_proba(X=test_X)  # この変数は出力されていないが、必要に応じて使用可能

# 予測されたターゲットと正解のターゲットの出力
print("Predicted Targets:", y_pred_classes)
print("True Targets:", test_Y)


# 評価
accuracy = accuracy_score(test_Y, y_pred_classes)
print(f"Accuracy: {accuracy * 100:.2f}%")

f1 = f1_score(test_Y, y_pred_classes, average='weighted')
print(f"F1 Score: {f1:.2f}")

