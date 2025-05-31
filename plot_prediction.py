from Li_metal_feature_extraction import *
from sklearn.model_selection import LeaveOneOut
from sklearn.pipeline import make_pipeline
from sklearn.linear_model import LinearRegression, ElasticNet
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import cross_validate
from sklearn.model_selection import cross_val_predict
from sklearn.dummy import DummyRegressor
from sklearn.metrics import mean_squared_error
from sklearn.metrics import mean_absolute_percentage_error
from sklearn import metrics as met
from sklearn.ensemble import RandomForestRegressor
import matplotlib.pyplot as plt

rmse_mean, rmse_std = [], []
true_rmse_mean, true_rmse_std = [], []
mape_mean, mape_std = [], []
true_mape_mean, true_mape_std = [], []

def true_RMSE(y_true, y_pred):
    return mean_squared_error(np.power(10, y_true), np.power(10, y_pred), squared=False)

def true_MAPE(y_true, y_pred):
    return mean_absolute_percentage_error(np.power(10, y_true), np.power(10, y_pred))

#eof: type of end of life condition: 
# '80': reach 80%
# 'short' short circut
# 'all': full dataset
np.random.seed(42)
Xs, Ys = get_Li_metal_all_feature_dataset(eof='short')
p = np.random.permutation(len(Xs))
Xs = Xs[p]
Ys = Ys[p]

trainlen = int(0.8*len(Xs))
train_Xs = Xs[:trainlen]
train_Ys = Ys[:trainlen]
val_Xs = Xs[trainlen:]
val_Ys = Ys[trainlen:]

scaler = StandardScaler().fit(train_Xs)
X_train_scaled = scaler.transform(train_Xs)
X_val_scaled = scaler.transform(val_Xs)

enet=ElasticNet()
enet.fit(X_train_scaled, train_Ys)

y_train_pred = enet.predict(X_train_scaled)
y_val_pred = enet.predict(X_val_scaled)

fig = plt.figure(figsize=(10,10))
ax1 = fig.add_subplot(111)

ax1.scatter(train_Ys, y_train_pred, c='crimson',label='Train')
ax1.scatter(val_Ys, y_val_pred, c='blue',label='Validation')

p1 = max(max(y_train_pred), max(train_Ys))
p2 = min(min(y_train_pred), min(train_Ys))
ax1.plot([250, 30], [250, 30], 'b-')
ax1.legend()
plt.xlabel('True Values', fontsize=15)
plt.ylabel('Predictions', fontsize=15)
plt.axis('equal')
plt.savefig('plt/Pred_True_short_best.png')