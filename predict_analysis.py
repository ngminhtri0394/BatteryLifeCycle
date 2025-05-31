import numpy as np
from sklearn import preprocessing
# from feature_extraction import *
from feature_extraction import *
from sklearn.model_selection import LeaveOneOut
from sklearn.pipeline import make_pipeline
from sklearn.linear_model import LinearRegression,ElasticNet
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import cross_validate

#All features
# Xs, Ys = get_dataset()
# DeltaQ Variance feature
# Xs, Ys = get_dataset_variance()
# DeltaQ feature
# Xs, Ys = get_dataset_discharge()
# DeltaQ + Capacity curve feature

rmse_mean, rmse_std = [], []
mape_mean, mape_std = [], []
for end in [25]:
    Xs, Ys = get_dataset_full_with_EIS(start=15, end=end)
    loo = LeaveOneOut()

    pipe = make_pipeline(StandardScaler(), ElasticNet())
    scoring = {'perror': 'neg_mean_absolute_percentage_error',
               'rmse': 'neg_root_mean_squared_error'}
    scores = cross_validate(pipe, Xs, Ys, cv=loo, scoring=scoring)

    rmse_mean.append(np.mean(scores['test_rmse']))
    rmse_std.append(np.std(scores['test_rmse']))

    mape_mean.append(np.mean(scores['test_perror']))
    mape_std.append(np.std(scores['test_perror']))

    # print(scores['test_perror'])

print('Negative RMSE: %.3f (%.3f)' % (np.mean(rmse_mean), np.mean(rmse_std)))
print('Negative MAPE: %.3f (%.3f)' % (np.mean(mape_mean), np.mean(mape_std)))
# print('Standard deviation of MAPE:', np.std(mape_mean))
