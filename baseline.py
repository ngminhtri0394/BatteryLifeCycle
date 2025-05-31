import pandas as pd
import numpy as np
from feature_extraction import *
from sklearn.model_selection import LeaveOneOut
from sklearn.pipeline import make_pipeline
from sklearn.linear_model import LinearRegression,ElasticNet
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import cross_validate


def get_dataset_full(start, end):
    df_label = pd.read_csv('BatteryData/clean_label.csv')

    Xs = np.array([])
    Ys = np.array([])
    cellsid = df_label['battery'].to_list()
    for cell in cellsid:
        filename_discharge = 'BatteryData/Cell ' + str(cell) + ' - Discharge.csv'
        df = pd.read_csv(filename_discharge)
        df = df.groupby(['cycle number'], sort=False)['Capacity/mA.h'].max().reset_index()
        df = df[(df['cycle number'] >= start) & (df['cycle number'] <= end)]
        feature = df['Capacity/mA.h'].values
        Xs = np.append(Xs, feature)
        Ys = np.append(Ys, df_label.loc[df_label['battery'] == cell]['life cycle'])

    num_samples = 26
    Xs = Xs.reshape(num_samples, -1)
    return Xs, Ys


if __name__ == '__main__':
    rmse_mean, rmse_std = [], []
    mape_mean, mape_std = [], []
    # for end in [25, 30, 35, 40, 45, 50]:
    for end in [25]:
        Xs, Ys = get_dataset_full(start=15, end=end)
        loo = LeaveOneOut()

        pipe = make_pipeline(StandardScaler(), ElasticNet())
        scoring = {'perror': 'neg_mean_absolute_percentage_error',
                   'rmse': 'neg_root_mean_squared_error'}
        scores = cross_validate(pipe, Xs, Ys, cv=loo, scoring=scoring)

        rmse_mean.append(np.mean(scores['test_rmse']))
        rmse_std.append(np.std(scores['test_rmse']))

        mape_mean.append(np.mean(scores['test_perror']))
        mape_std.append(np.std(scores['test_perror']))

    print('Negative RMSE: %.3f (%.3f)' % (np.mean(rmse_mean), np.mean(rmse_std)))
    print('Negative MAPE: %.3f (%.3f)' % (np.mean(mape_mean), np.mean(mape_std)))
    print('Standard deviation of MAPE:', np.std(mape_mean))
