
from sktime.datasets import load_airline
from sktime.forecasting.model_selection import temporal_train_test_split
import matplotlib.pyplot as plt
from pyrcn.echo_state_network import ESNRegressor
from sklearn.metrics import mean_squared_error
from math import sqrt

'''
sktimeに含まれるairlineデータセットの読み込み
'''
y = load_airline()
print(type(y))

#データセットの概要
print(y.describe())

# Series オブジェクトをプロット

plt.plot(y.values)
plt.xlabel('month')
plt.ylabel('Passengers')
plt.title('Airline Dataset')
plt.savefig('airline_test_data_plot.png',dpi=600)
plt.show()





#データの分裂
y_train, y_test = temporal_train_test_split(y)

#学習データのプロット
'''
plt.plot(y_train.values)
plt.xlabel('Time')
plt.ylabel('Passengers')
plt.title('Train Data')
plt.show()
'''


#テストデータのプロット
'''
plt.plot(y_test.values)
plt.xlabel('Time')
plt.ylabel('Passengers')
plt.title('Train Data')
plt.show()
'''


#モデルの構築
# Vanilla ESN for regression tasks with spectral_radius and leakage
#       _ _ _ _ _ _ _       _ _ _ _ _ _ _        _ _ _ _ _ _ _        
#     |              |     |             |     |               |       
# ----|Input-to-Node |-----|Node-to-Node |-----|Node-to-Output |
# u[n]| _ _ _ _ _ _ _|r'[n]|_ _ _ _ _ _ _|r[n] | _ _ _ _ _ _ _ |
#                                                      |
#                                                      |
#                                                 y[n] | y_pred
#spectral_radius : リザバー同士の結びつきの強さを決める役割
#leaking_rate : 前のからの隠れ状態をどれだけ次の隠れ状態に影響を与えるかを制御する役割
base_esn = ESNRegressor(spectral_radius=0.6, leakage=0.8)

# データの形式を変換
y_train_values = y_train.values.reshape(-1, 1)
y_test_values = y_test.values.reshape(-1, 1)

# モデルの学習
base_esn.fit(y_train_values, y_train_values)
print('モデルの学習完了！')


# モデルの予測
y_pred =base_esn.predict(y_test_values)

# 予測結果のプロット
# データセット全体のプロット
plt.plot(y.values, label='Original')
# 予測結果のプロット
plt.plot(range(len(y_train), len(y)), y_pred, label='Predicted', linestyle='dashed')
plt.xlabel('Month')
plt.ylabel('Passengers')
plt.title('Airline Dataset with Prediction')
plt.legend()
plt.savefig('airline_test_data_prediction_plot.png', dpi=600)
plt.show()


# MSEの計算
rmse = sqrt(mean_squared_error(y_test.values, y_pred))
print('Root Mean Squared Error: {:.4f}'.format(rmse))