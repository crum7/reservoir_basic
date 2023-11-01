import pickle
import numpy as np
from sktime.datasets import load_italy_power_demand
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, f1_score, make_scorer, mean_squared_error
from sklearn.model_selection import TimeSeriesSplit, GridSearchCV, RandomizedSearchCV
from scipy.stats import uniform
from pyrcn.echo_state_network import ESNClassifier
from pyrcn.model_selection import SequentialSearchCV
import numpy as np
from sklearn.model_selection import cross_val_score
from pyrcn.echo_state_network import ESNClassifier
import optuna

'''
論文のモデル実装のESN部分
Reservoir Computing with Untrained Convolutional Neural Networks for Image Recognition
https://ieeexplore.ieee.org/stamp/stamp.jsp?tp=&arnumber=8545471&tag=1

'''

# feature_maps.pklから特徴マップを読み出す


# 特徴マップを読み込む
feature_maps_train = np.load('feature_maps_train.npy')
feature_maps_test = np.load('feature_maps_test.npy')

# ラベルを読み込む
y_train = np.load('labels_train.npy')
y_test = np.load('labels_test.npy')



# xを2次元に変換
#元々は、(10000, 2, 2, 84)の形なんだけど、それを(10000,2*2*84)に変換する。
#10000は、MNISTのtestのデータ数
feature_maps_train = feature_maps_train.reshape(60000, -1)
feature_maps_test = feature_maps_test.reshape(10000, -1)

# データの分割
# データの分割
train_X=feature_maps_train
test_X = feature_maps_test
train_Y = y_train
test_Y = y_test

print("train_X shape:", train_X.shape)
print("train_Y shape:", train_Y.shape)



def objective(trial):
    input_scaling = trial.suggest_float('input_scaling', 0, 1)
    hidden_layer_size = trial.suggest_int('hidden_layer_size', 1000, 6000)
    spectral_radius = trial.suggest_float('spectral_radius', 0, 1)
    leaking_rate = trial.suggest_float('leaking_rate', 0, 1)
    sparsity = trial.suggest_float('sparsity', 0, 1)


    esn = ESNClassifier(
        input_scaling=input_scaling,
        hidden_layer_size=hidden_layer_size,
        spectral_radius=spectral_radius,
        leaking_rate=leaking_rate,
        sparsity=sparsity,
        reservoir_activation='tanh',
        decision_strategy='vote',
    )

    # モデルのフィッティング
    esn.fit(train_X, train_Y)


    # テストデータに対する予測
    test_y_pred = esn.predict(test_X)
    # トレインデータに対する予測
    train_y_pred = esn.predict(train_X)
    # 評価
    test_accuracy = accuracy_score(test_Y, test_y_pred)
    train_accuracy = accuracy_score(train_Y, train_y_pred)

    test_f1 = f1_score(test_Y, test_y_pred, average='weighted')
    train_f1 = f1_score(train_Y, train_y_pred, average='weighted')

    print('Test Accuracy:', test_accuracy)
    print('Train Accuracy:', train_accuracy)
    print('Test F1 Score (weighted):', test_f1)
    print('Train F1 Score (weighted):', train_f1)

    return cross_val_score(esn, train_X, train_Y, n_jobs=-1, cv=3).mean()

# Optunaのスタディを作成し、目的関数を最適化
study = optuna.create_study(direction='maximize')
study.optimize(objective, n_trials=300)

# 最適なハイパーパラメータの表示
print('Number of finished trials: ', len(study.trials))
print('Best trial:')
trial = study.best_trial
print('Value: ', trial.value)
print('Params: ')
for key, value in trial.params.items():
    print(f'    {key}: {value}')
