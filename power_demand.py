import numpy as np
from sktime.datasets import load_italy_power_demand
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, f1_score, make_scorer, mean_squared_error
from sklearn.model_selection import TimeSeriesSplit, GridSearchCV, RandomizedSearchCV
from scipy.stats import uniform
from pyrcn.echo_state_network import ESNClassifier
from pyrcn.model_selection import SequentialSearchCV
import matplotlib.pyplot as plt



# データの読み込み・前処理
x, y = load_italy_power_demand()
x = np.array([row.tolist() for row in x['dim_0']])
y = y.astype(int).ravel()-1
print(type(x))
print(type(y))
#各時系列データの長さを知る。
lengths = [len(data) for data in x]
print(lengths)



# データの分割
train_X, test_X, train_Y, test_Y = train_test_split(x, y, train_size=0.6, random_state=0)



# ハイパーパラメータの探索
scorer = make_scorer(accuracy_score, greater_is_better=True)
step_1_params = {
    'input_scaling': uniform(0, 1),
    'hidden_layer_size': [30,35,40,50,60],
    'spectral_radius': uniform(0, 1.5),
    'leaking_rate': uniform(0, 1),
}

#n_iter:パラメータの組み合わせを選ぶ際の試行回数
#cv : Cross Validation（交差検証）の分割数

kwargs_1 = {
    'n_iter': 300, 'n_jobs': -1, 'scoring': scorer,'cv': 3
}

searches = [('step1', RandomizedSearchCV, step_1_params, kwargs_1)]


#ハーパーパラメータの検索
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

# オリジナルデータのプロット
plt.figure(figsize=(10, 5))
for i, sample in enumerate(test_X[:5]):
    plt.plot(sample, label=f"True Class: {test_Y[i]}")

# 推論結果のプロット
for i, sample in enumerate(test_X[:5]):
    plt.plot(sample, label=f"Predicted Class: {y_pred_classes[i]}", linestyle='dashed')

plt.xlabel('Time')
plt.ylabel('Value')
plt.title('Original Data and Predicted Results')
plt.legend()
plt.show()


