# from feature_extraction import *
# from EIS_feature_extraction import *
# from nature_feature_extraction import *
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
from sklearn.gaussian_process import GaussianProcessRegressor

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
# 
#feature: nature paper
Xs, Ys = get_Li_metal_all_feature_dataset(eof='80')
Xtest, Ytest = get_Li_metal_test_cell()

# feature: EIS + variance of average discharge voltage
# Xs, Ys = get_Li_metal_EIS_AvgV1_dataset(eof='80')

loo = LeaveOneOut()
# true_RMSE = met.make_scorer(true_RMSE, greater_is_better = False)
# true_MAPE = met.make_scorer(true_MAPE, greater_is_better = False)

model = ElasticNet()

scaler = StandardScaler().fit(Xs)
X_train_scaled = scaler.transform(Xs)
X_test_scaled = scaler.transform(Xtest)

model.fit(X_train_scaled, Ys)
ypred = model.predict(X_test_scaled)
print(ypred)
print(Ytest)

rmse = mean_squared_error(ypred,Ytest,squared=False)
mape = mean_absolute_percentage_error(ypred,Ytest)
print(rmse)
print(mape)