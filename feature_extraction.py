import pandas as pd
import matplotlib.pyplot as plt
import glob
import numpy as np
from derivative import CurveDerivative
from EIS_feature_extraction import get_EIS_feature


def get_all_max_capacities(df, start, end):
    df = df.groupby(['cycle number'], sort=False)['Capacity/mA.h'].max().reset_index()
    df = df[(df['cycle number'] >= start) & (df['cycle number'] <= end)]
    capacity = df['Capacity/mA.h'].to_list()
    return np.array(capacity)


def get_dataset_full(start, end):
    df_label = pd.read_csv('BatteryData/clean_label.csv')

    Xs = np.array([])
    Ys = np.array([])
    cellsid = df_label['battery'].to_list()
    for cell in cellsid:
        feature = np.array([])

        # Get discharge data
        filename_discharge = 'BatteryData/Cell ' + str(cell) + ' - Discharge.csv'
        df_discharge = pd.read_csv(filename_discharge)
        start=df_discharge['cycle number'].min()
        end = start +10

        # Get all max capacities
        capacity = get_all_max_capacities(df_discharge, start=start, end=end)
        feature = np.append(feature, capacity)

        # Feature of curve
        derivative = CurveDerivative(order=0, alpha=1, beta=1)
        deltaQ_feature = derivative.get_deltaQ_feature(df=df_discharge, start=start, end=end,
                                                       feature_list=['max', 'logvar'])
        feature = np.append(feature, deltaQ_feature)

        # Feature of curve
        derivative = CurveDerivative(order=0, alpha=0.5, beta=4)
        deltaQ_feature = derivative.get_deltaQ_feature(df=df_discharge, start=start, end=end,
                                                       feature_list=['logskew'])
        feature = np.append(feature, deltaQ_feature)

        # Feature of 1st derivative
        derivative = CurveDerivative(order=1, alpha=3.5, beta=0.5)
        deltaQ_feature = derivative.get_deltaQ_feature(df=df_discharge, start=start, end=end,
                                                       feature_list=['logvar', 'logskew'])
        feature = np.append(feature, deltaQ_feature)

        # Feature of 2nd derivative
        derivative = CurveDerivative(order=2, alpha=0.25, beta=0.5)
        deltaQ_feature = derivative.get_deltaQ_feature(df=df_discharge, start=start, end=end,
                                                       feature_list=['logvar', 'logkur'])
        feature = np.append(feature, deltaQ_feature)

        # Feature of 3rd derivative
        derivative = CurveDerivative(order=3, alpha=0.25, beta=0.75)
        deltaQ_feature = derivative.get_deltaQ_feature(df=df_discharge, start=start, end=end,
                                                       feature_list=['logvar', 'logskew', 'logkur', 'max'])
        feature = np.append(feature, deltaQ_feature)

        # Feature of 5th derivative
        derivative = CurveDerivative(order=5, alpha=1, beta=1)
        deltaQ_feature = derivative.get_deltaQ_feature(df=df_discharge, start=start, end=end,
                                                       feature_list=['logkur'])
        feature = np.append(feature, deltaQ_feature)

        Xs = np.append(Xs, feature)
        Ys = np.append(Ys, df_label.loc[df_label['battery'] == cell]['life cycle'])

    num_samples = 26
    Xs = np.reshape(Xs, (num_samples, -1))
    return Xs, Ys


def get_dataset_full_with_EIS(start, end):
    df_label = pd.read_csv('BatteryData/clean_label.csv')

    Xs = np.array([])
    Ys = np.array([])
    cellsid = df_label['battery'].to_list()
    for cell in cellsid:
        feature = np.array([])

        # Get discharge data
        filename_discharge = 'BatteryData/Cell ' + str(cell) + ' - Discharge.csv'
        df_discharge = pd.read_csv(filename_discharge)
        start = df_discharge['cycle number'].min()
        end = start + 10

        # Get all max capacities
        capacity = get_all_max_capacities(df_discharge, start=start, end=end)
        feature = np.append(feature, capacity)

        # Feature of curve
        derivative = CurveDerivative(order=0, alpha=1, beta=1)
        deltaQ_feature = derivative.get_deltaQ_feature(df=df_discharge, start=start, end=end,
                                                       feature_list=['max', 'logvar'])
        feature = np.append(feature, deltaQ_feature)

        # Feature of curve
        derivative = CurveDerivative(order=0, alpha=0.5, beta=4)
        deltaQ_feature = derivative.get_deltaQ_feature(df=df_discharge, start=start, end=end,
                                                       feature_list=['logskew'])
        feature = np.append(feature, deltaQ_feature)

        # Feature of 1st derivative
        derivative = CurveDerivative(order=1, alpha=3.5, beta=0.5)
        deltaQ_feature = derivative.get_deltaQ_feature(df=df_discharge, start=start, end=end,
                                                       feature_list=['logvar', 'logskew'])
        feature = np.append(feature, deltaQ_feature)

        # Feature of 2nd derivative
        derivative = CurveDerivative(order=2, alpha=0.25, beta=0.5)
        deltaQ_feature = derivative.get_deltaQ_feature(df=df_discharge, start=start, end=end,
                                                       feature_list=['logvar', 'logkur'])
        feature = np.append(feature, deltaQ_feature)

        # Feature of 3rd derivative
        derivative = CurveDerivative(order=3, alpha=0.25, beta=0.75)
        deltaQ_feature = derivative.get_deltaQ_feature(df=df_discharge, start=start, end=end,
                                                       feature_list=['logvar', 'logskew', 'logkur', 'max'])
        feature = np.append(feature, deltaQ_feature)

        # Feature of 5th derivative
        derivative = CurveDerivative(order=5, alpha=1, beta=1)
        deltaQ_feature = derivative.get_deltaQ_feature(df=df_discharge, start=start, end=end,
                                                       feature_list=['logkur'])
        feature = np.append(feature, deltaQ_feature)

        EISfeature = get_EIS_feature(cell,9990)
        feature = np.append(feature, EISfeature)

        Xs = np.append(Xs, feature)
        Ys = np.append(Ys, df_label.loc[df_label['battery'] == cell]['life cycle'])

    num_samples = 26
    Xs = np.reshape(Xs, (num_samples, -1))
    return Xs, Ys
