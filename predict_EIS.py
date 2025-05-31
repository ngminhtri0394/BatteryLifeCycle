from Li_metal_feature_extraction import *
from sklearn.model_selection import LeaveOneOut, RepeatedKFold
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

rmse_mean, rmse_std = [], []
true_rmse_mean, true_rmse_std = [], []
mape_mean, mape_std = [], []
true_mape_mean, true_mape_std = [], []

def true_RMSE(y_true, y_pred):
    return mean_squared_error(np.power(10, y_true), np.power(10, y_pred), squared=False)

def true_MAPE(y_true, y_pred):
    return mean_absolute_percentage_error(np.power(10, y_true), np.power(10, y_pred))

Xs, Ys = get_Li_metal_all_feature_dataset(eof='all')


loo = LeaveOneOut()
rkf = RepeatedKFold(n_splits=5, n_repeats=5, random_state=42)

model = ElasticNet(max_iter=10000)


pipe = make_pipeline(StandardScaler(), model)
scoring = {'perror': 'neg_mean_absolute_percentage_error',
           'rmse': 'neg_root_mean_squared_error',}

scores = cross_validate(pipe, Xs, Ys, cv=loo, scoring=scoring)

rmse_mean.append(np.mean(scores['test_rmse']))
rmse_std.append(np.std(scores['test_rmse']))

mape_mean.append(np.mean(scores['test_perror']))
mape_std.append(np.std(scores['test_perror']))


print('Negative RMSE: %.3f (%.3f)' % (np.mean(rmse_mean), np.mean(rmse_std)))
print('Negative MAPE: %.3f (%.3f)' % (np.mean(mape_mean), np.mean(mape_std)))
