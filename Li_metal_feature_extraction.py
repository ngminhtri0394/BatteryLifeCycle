import pandas as pd
import matplotlib.pyplot as plt
import glob
import numpy as np
from scipy import interpolate
from scipy.interpolate import splev, splrep
from scipy.stats import skew,kurtosis
# from EIS_feature_extraction import get_EIS_feature
from impedance.models.circuits import CustomCircuit
from impedance import preprocessing
from sklearn import preprocessing

def delta_discharge_capacity_curve(df, i, j,vset):
    df_cycle_i = df.loc[df['cycle number'] == i].sort_values(by=['Ecell/V'])
    df_cycle_i = df_cycle_i.drop_duplicates(subset=['Ecell/V'])
    v_i = df_cycle_i['Ecell/V'].tolist()
    q_i = df_cycle_i['Capacity/mA.h'].tolist()
    spl_i = splrep(v_i, q_i)
    new_qi = splev(vset, spl_i)

    df_cycle_j = df.loc[df['cycle number'] == j].sort_values(by=['Ecell/V'])
    df_cycle_j = df_cycle_j.drop_duplicates(subset=['Ecell/V'])
    v_j = df_cycle_j['Ecell/V'].tolist()
    q_j = df_cycle_j['Capacity/mA.h'].tolist()
    spl_j = splrep(v_j, q_j)
    new_qj = splev(vset, spl_j)

    delta_q = abs(np.array(new_qj)-np.array(new_qi))
    return delta_q

def get_full_EIS_interpolation(freqstart,freqend,freq, rez, imz, mZ, pZ):
    freqset = np.linspace(freqstart, freqend, num=30)
    spl_i_rez = splrep(freq, rez)
    spl_i_imz = splrep(freq, imz)
    spl_i_mZ = splrep(freq, mZ)
    spl_i_pZ = splrep(freq, pZ)

    new_rez = splev(freqset, spl_i_rez)
    new_imz = splev(freqset, spl_i_imz)
    new_mZ = splev(freqset, spl_i_mZ)
    new_pz = splev(freqset, spl_i_pZ)
    return new_rez, new_imz, new_mZ, new_pz


def get_deltaQ_feature(df, start, end):
    vset = np.linspace(2.8, 3.42, 1000)
    deltaq = delta_discharge_capacity_curve(df, start, end ,vset)
    logvar = np.log10(np.var(deltaq))
    skewness = np.log10(abs(skew(deltaq)))
    kur = np.log10(abs(kurtosis(deltaq)))
    return np.array([max(deltaq),logvar, skewness, kur])


def get_discharge_fade_curve_feature(df, start, end):
    df_cycle = df.groupby(['cycle number'], sort=False)['Capacity/mA.h'].max().reset_index()
    df_cycle = df_cycle[(df_cycle['cycle number'] >= start) & (df_cycle['cycle number'] <= end)]
    cap_curve = df_cycle['Capacity/mA.h'].to_list()
    log_cycle_number = np.log10(df_cycle['cycle number'].to_list())
    b = np.polyfit(log_cycle_number, cap_curve, deg=1)
    return np.array([b[0], b[1], df.loc[df['cycle number'] == start]['Capacity/mA.h'].max()])


def get_cell_id_from_failure_mode(eof='all'):
    df_label = pd.read_csv('BatteryData/Li_metal_label.csv')
    if eof == 'all':
        cellsid = df_label['battery'].values.tolist()
    elif eof == 'short':
        cellsid = df_label[df_label['Failure mode']=='Short circuit']['battery'].values.tolist()
    else:
        cellsid = df_label[df_label['Failure mode']=='80%']['battery'].values.tolist()

    return cellsid


def get_Li_metal_deltaQ_dataset(eof='all'):
    cellsid = get_cell_id_from_failure_mode(eof=eof)
    print(cellsid)
    df_label = pd.read_csv('BatteryData/Li_metal_label.csv')
    Xs = np.array([])
    Ys = np.array([])
    for cell in cellsid:
        filename = 'BatteryData/Cell ' + str(cell) + ' - Discharge.csv'
        df = pd.read_csv(filename)
        start = df['cycle number'].min()
        end = start + 10
        discharge_curve_feature = get_discharge_fade_curve_feature(df, start, end)
        deltaQ_feature = get_deltaQ_feature(df, start, end)
        feature = np.append(discharge_curve_feature, deltaQ_feature)
        Xs = np.append(Xs,feature)
        Ys = np.append(Ys, np.log10(df_label.loc[df_label['battery'] == cell]['Corrected cycle life']))
    return np.reshape(Xs,(len(cellsid),-1)), Ys


def get_EIS_interpolation(freqstart,freqend,freq, rez,imz):
    freqset = np.linspace(freqstart, freqend, num=30)
    spl_i_rez = splrep(freq, rez)
    spl_i_imz = splrep(freq, imz)
    new_rez = splev(freqset, spl_i_rez)
    new_imz = splev(freqset, spl_i_imz)
    return new_rez, new_imz


def get_deltaEIS_feature(cell_EIS):
    min_EIS_cycle_number = cell_EIS['cycle number'].min()
    max_EIS_cycle_number = cell_EIS['cycle number'].max()

    EIScycle = cell_EIS.loc[cell_EIS['cycle number'] == min_EIS_cycle_number]
    EIScycle = EIScycle.drop_duplicates(subset=['freq/Hz'])
    freq = EIScycle['freq/Hz'].to_list()
    reZ = EIScycle['Re(Z)/Ohm'].to_list()
    imZ = EIScycle['-Im(Z)/Ohm'].to_list()

    sort_reZ = [x for _, x in sorted(zip(freq, reZ))]
    sort_imZ = [x for _, x in sorted(zip(freq, imZ))]

    new_rez_start, new_imz_start = get_EIS_interpolation(1, 9990, sorted(freq), sort_reZ, sort_imZ)

    EIScycle = cell_EIS.loc[cell_EIS['cycle number'] == max_EIS_cycle_number]
    EIScycle = EIScycle.drop_duplicates(subset=['freq/Hz'])
    freq = EIScycle['freq/Hz'].to_list()
    reZ = EIScycle['Re(Z)/Ohm'].to_list()
    imZ = EIScycle['-Im(Z)/Ohm'].to_list()

    sort_reZ = [x for _, x in sorted(zip(freq, reZ))]
    sort_imZ = [x for _, x in sorted(zip(freq, imZ))]
    new_rez_end, new_imz_end = get_EIS_interpolation(1, 9990, sorted(freq), sort_reZ, sort_imZ)

    delta_rez = abs(np.array(new_rez_start)-np.array(new_rez_end))
    delta_imz = abs(np.array(new_imz_start)-np.array(new_imz_end))

    var_delta_rez = np.var(delta_rez)
    var_delta_imz = np.var(delta_imz)

    return delta_rez, var_delta_rez, delta_imz, var_delta_imz


def get_EIS_feature(cellid,freqlb=1,frequb=9990):
    cell_EIS = pd.read_csv('BatteryData/Cell ' + str(cellid) + ' - EIS.csv')
    min_EIS_cycle_number = cell_EIS['cycle number'].min()

    EIScycle = cell_EIS.loc[cell_EIS['cycle number'] == min_EIS_cycle_number]
    EIScycle = EIScycle.drop_duplicates(subset=['freq/Hz'])
    freq = EIScycle['freq/Hz'].to_list()
    reZ = EIScycle['Re(Z)/Ohm'].to_list()
    imZ = EIScycle['-Im(Z)/Ohm'].to_list()

    sort_reZ = [x for _, x in sorted(zip(freq, reZ))]
    sort_imZ = [x for _, x in sorted(zip(freq, imZ))]

    new_rez, new_imz = get_EIS_interpolation(freqlb, frequb, sorted(freq), sort_reZ, sort_imZ)
    X = np.ravel([new_rez, new_imz], 'F')
    return X

def get_Li_metal_EIS_full_dataset(cellid,freqlb=1,frequb=9990):
    cell_EIS = pd.read_csv('BatteryData/Cell '+str(cellid) + ' - EIS_full.csv')
    min_EIS_cycle_number = cell_EIS['cycle number'].min()
    max_EIS_cycle_number = cell_EIS['cycle number'].max()

    # single EIS cycle feature
    EIScycle = cell_EIS.loc[cell_EIS['cycle number'] == min_EIS_cycle_number]
    EIScycle = EIScycle.drop_duplicates(subset=['freq/Hz'])
    freq = EIScycle['freq/Hz'].to_list()[:-1]
    reZ = EIScycle['Re(Z)/Ohm'].to_list()[:-1]
    imZ = EIScycle['-Im(Z)/Ohm'].to_list()[:-1]
    mZ = EIScycle['|Z|/Ohm'].to_list()[:-1]
    pZ = EIScycle['Phase(Z)/deg'].to_list()[:-1]

    sort_reZ = [x for _, x in sorted(zip(freq, reZ))]
    sort_imZ = [x for _, x in sorted(zip(freq, imZ))]
    sort_mZ = [x for _, x in sorted(zip(freq, mZ))]
    sort_pZ = [x for _, x in sorted(zip(freq, pZ))]

    new_rez, new_imz, new_mZ, new_pz = get_full_EIS_interpolation(freqlb, frequb, sorted(freq), sort_reZ, sort_imZ, sort_mZ, sort_pZ)
    X = np.ravel([new_rez, new_imz, new_mZ, new_pz],'F')
    return X

def get_Li_metal_EIS_dataset(eof='all', freqlb=1, frequb=9990):
    cellsid = get_cell_id_from_failure_mode(eof=eof)
    df_label = pd.read_csv('BatteryData/Li_metal_label.csv')
    Xs = np.array([])
    Ys = np.array([])
    for cell in cellsid:
        if cell == 18 or cell == 9 or cell == 21 or cell == 19:
            continue
        EIS_feature = get_EIS_feature(cell, freqlb=freqlb, frequb=frequb) # reported
        # EIS_feature = get_Li_metal_EIS_full_dataset(cell, freqlb=freqlb, frequb=frequb) # reported
        cell_life_cycle = df_label.loc[df_label['battery']==cell]['Corrected cycle life']
        Xs = np.append(Xs, EIS_feature)
        Ys = np.append(Ys, cell_life_cycle)
    Xs = np.reshape(Xs, (len(cellsid)-2, -1))
    return Xs, Ys

def get_Li_metal_Avg_EIS_dataset(eof='all'):
    cellsid = get_cell_id_from_failure_mode(eof=eof)
    df_label = pd.read_csv('BatteryData/Li_metal_label.csv')
    Xs = np.array([])
    Ys = np.array([])
    for cell in cellsid:
        avg_realZ_10 =  df_label.loc[df_label['battery']==cell]['Avg. of Real (Z) (10 cycles)'].values
        avg_imZ_10 =  df_label.loc[df_label['battery']==cell]['Avg. of Img (Z) (10 cycles)'].values
        cell_life_cycle = df_label.loc[df_label['battery']==cell]['Corrected cycle life']
        Xs = np.append(Xs, [avg_realZ_10,avg_imZ_10])
        Ys = np.append(Ys, cell_life_cycle)
    Xs = np.reshape(Xs, (len(cellsid), -1))
    return Xs, Ys

def get_Li_metal_avg_charge_vol_dataset(eof='all'):
    cellsid = get_cell_id_from_failure_mode(eof=eof)
    df_label = pd.read_csv('BatteryData/Li_metal_label.csv')
    Xs = np.array([])
    Ys = np.array([])
    for cell in cellsid:
        avg_ch_vol =  df_label.loc[df_label['battery']==cell]['Average charge voltage (1st cycle)'].values
        avg_chg_vol_15_1 = df_label.loc[df_label['battery']==cell]['Average charge voltage (first 15% of capacity) - 1st cycle'].values
        var_avg_ch_vol_10 =  df_label.loc[df_label['battery']==cell]['Var of avg. chg. Voltage (10 cycles)'].values

        cell_life_cycle = df_label.loc[df_label['battery']==cell]['Corrected cycle life']
        Xs = np.append(Xs, [avg_chg_vol_15_1])
        Ys = np.append(Ys, cell_life_cycle)
    Xs = np.reshape(Xs, (len(cellsid), -1))
    return Xs, Ys

def get_Li_metal_avg_charge_vol_EIS_dataset(eof='all'):
    cellsid = get_cell_id_from_failure_mode(eof=eof)
    df_label = pd.read_csv('BatteryData/Li_metal_label.csv')
    Xs = np.array([])
    Ys = np.array([])
    for cell in cellsid:
        avg_ch_vol =  df_label.loc[df_label['battery']==cell]['Average charge voltage (1st cycle)'].values[0]
        avg_chg_vol_15_1 = df_label.loc[df_label['battery']==cell]['Average charge voltage (first 15% of capacity) - 1st cycle'].values[0]
        var_avg_ch_vol_10 =  df_label.loc[df_label['battery']==cell]['Var of avg. chg. Voltage (10 cycles)'].values[0]
        EIS_feature = get_EIS_feature(cell)
        cell_life_cycle = df_label.loc[df_label['battery']==cell]['Corrected cycle life']
        avg_chg_vol_feat = [var_avg_ch_vol_10,avg_ch_vol,avg_chg_vol_15_1]
        avg_chg_vol_feat.extend(EIS_feature)
        Xs = np.append(Xs, avg_chg_vol_feat)
        Ys = np.append(Ys, cell_life_cycle)
    Xs = np.reshape(Xs, (len(cellsid), -1))
    return Xs, Ys

def get_features_df(df, feature_list, cell):
    return df.loc[df['battery']==cell][feature_list].values.astype(np.float).tolist()[0]

def get_all_features_df(df, cell):
    df.loc[df['battery']==cell].values.astype(np.float).tolist()[0]

def get_delta_avg_charge_discharge_all_feature(df,cell):
    ADF1 = np.array(df.loc[df['battery']==cell]['ADF_15_1'].values.astype(np.float).tolist()[0])
    ADF10 = np.array(df.loc[df['battery']==cell]['ADF_15_10'].values.astype(np.float).tolist()[0])
    return ADF10 - ADF1

def get_Li_metal_avg_charge_discharge_all_feature(eof='all'):
    cellsid = get_cell_id_from_failure_mode(eof=eof)
    df_label = pd.read_csv('BatteryData/avg_voltage.csv')
    Xs = np.array([])
    Ys = np.array([])
    # feature_list = list(df_label)[1:-2]
    feature_list = []
    feature_type = ['ADF', 'ADB','ACF','ACB']
    # feature_type = ['ADF','ACF']
    # feature_type = ['ADB','ACB']
    fb = ['_5_10_','_5_15_','_10_15_']
    ff = ['_10_','_15_']

    for fea in feature_type:
        f = fea
        if fea[-1] == 'B':
            for rfb in fb:
                fe = f + rfb
                for i in range(1,11):
                    feat = fe + str(i)
                    feature_list.append(feat)
        elif fea[-1] == 'F':
            for rf in ff:
                fe = f + rf
                for i in range(1,11):
                    feat = fe + str(i)
                    feature_list.append(feat)
        
        
    print(feature_list)
    for cell in cellsid:
        # features = get_features_df(df_label, feature_list ,cell)
        features = get_delta_avg_charge_discharge_all_feature(df_label,cell)
        Xs = np.append(Xs, features)
        cell_life_cycle = df_label.loc[df_label['battery']==cell]['Corrected cycle life']
        Ys = np.append(Ys, cell_life_cycle)
    Xs = np.reshape(Xs, (len(cellsid), -1))
    print(Xs.shape)
    return Xs, Ys

def get_Li_metal_avg_charge_discharge_1stcycle_dataset(eof='all'):
    cellsid = get_cell_id_from_failure_mode(eof=eof)
    df_label = pd.read_csv('BatteryData/Li_metal_label.csv')
    Xs = np.array([])
    Ys = np.array([])
    for cell in cellsid:
        features = get_features_df(df_label,[
                                                'Average charge voltage (first 5% of capacity) - 1st cycle',
                                                'Average charge voltage (first 10% of capacity) - 1st cycle',
                                                'Average charge voltage (first 15% of capacity) - 1st cycle',
                                                'Average charge voltage (between 5% &15% SoC) - 1st cycle',
                                                'Average charge voltage (between 10% &15% SoC) - 1st cycle',
                                                'Average discharge voltage (first 15% of capacity) - 1st cycle',
                                                'Average discharge voltage (first 10% of capacity) - 1st cycle',
                                                'Average discharge voltage (first 5% of capacity) - 1st cycle',
                                                'Average discharge voltage (between 5% &15% SoC) - 1st cycle',
                                                'Average discharge voltage (between 10% &15% SoC) - 1st cycle',
                                                'Average discharge voltage (between 5% &10% SoC) - 1st cycle'
                                             ]
                                             , cell)
        # cell_EIS = pd.read_csv('BatteryData/Cell '+str(cell) + ' - EIS.csv')
        # delta_rez, var_delta_rez, delta_imz, var_delta_imz = get_deltaEIS_feature(cell_EIS)
        # EIS_feature = get_EIS_feature(cell)
        # features.extend(EIS_feature)
        # # print(var_delta_rez)
        # features.append(var_delta_rez)
        # features.extend(delta_imz)
        # features.append(var_delta_imz)
        # features.extend(delta_rez)
        Xs = np.append(Xs, features)
        cell_life_cycle = df_label.loc[df_label['battery']==cell]['Corrected cycle life']
        Ys = np.append(Ys, cell_life_cycle)
    Xs = np.reshape(Xs, (len(cellsid), -1))
    # print(Xs)
    return Xs, Ys

def get_Li_metal_all_feature_dataset(eof='all'):
    cellsid = get_cell_id_from_failure_mode(eof=eof)
    df_label = pd.read_csv('BatteryData/Li_metal_label.csv')
    Xs = np.array([])
    Ys = np.array([])
    for cell in cellsid:
        if cell == 18 or cell == 9 or cell == 21 or cell == 19:
            continue
        features = get_features_df(df_label,[
                                            #  'Var of avg. disch. Voltage (10 cycles)', #-
                                             'Average discharge voltage (1st cycle)',
                                            #  'Average charge voltage (10th cycle)',
                                            #  'Average discharge voltage (10th cycle)', #-
                                             'Average charge voltage (1st cycle)',
                                            #  'Var of avg. chg. Voltage (10 cycles)', #-
                                            #  'Avg. of Real (Z) (10 cycles)', #-
                                            #  'Avg. of Img (Z) (10 cycles)', #-
                                             'Average charge voltage (first 5% of capacity) - 1st cycle', #-
                                             'Average charge voltage (first 10% of capacity) - 1st cycle',
                                             'Average charge voltage (first 15% of capacity) - 1st cycle',
                                             'Average charge voltage (between 5% &15% SoC) - 1st cycle',
                                             'Average charge voltage (between 10% &15% SoC) - 1st cycle',
                                             'Average charge voltage (between 5% &10% SoC) - 1st cycle',
                                             'Average discharge voltage (first 15% of capacity) - 1st cycle',
                                             'Average discharge voltage (first 10% of capacity) - 1st cycle',
                                             'Average discharge voltage (first 5% of capacity) - 1st cycle', #-
                                             'Average discharge voltage (between 5% &15% SoC) - 1st cycle',
                                             'Average discharge voltage (between 10% &15% SoC) - 1st cycle',
                                             'Average discharge voltage (between 5% &10% SoC) - 1st cycle'
                                             ]
                                             , cell)
        cell_EIS = pd.read_csv('BatteryData/Cell '+str(cell) + ' - EIS.csv')
        delta_rez, var_delta_rez, delta_imz, var_delta_imz = get_deltaEIS_feature(cell_EIS)
        EIS_feature = get_EIS_feature(cell)
        features.extend(EIS_feature)
        # print(var_delta_rez)
        # features.append(var_delta_rez)
        # features.extend(delta_imz)
        # features.append(var_delta_imz)
        # features.extend(delta_rez)
        Xs = np.append(Xs, features)
        cell_life_cycle = df_label.loc[df_label['battery']==cell]['Corrected cycle life']
        Ys = np.append(Ys, cell_life_cycle)
    if eof=='all':
        Xs = np.reshape(Xs, (len(cellsid)-4, -1))
    else:
        Xs = np.reshape(Xs, (len(cellsid)-2, -1))
    print(Xs.shape)
    return Xs, Ys

def get_Li_metal_test_cell():
    df_label = pd.read_csv('BatteryData/test_cells.csv')
    cellsid = df_label['battery'].values.tolist()
    Xs = np.array([])
    Ys = np.array([])
    for cell in cellsid:
        features = get_features_df(df_label,[
                                            #  'Var of avg. disch. Voltage (10 cycles)', #-
                                             'Average discharge voltage - 1st cycle',
                                            #  'Average discharge voltage (10th cycle)', #-
                                             'Average charge voltage - 1st cycle',
                                            #  'Var of avg. chg. Voltage (10 cycles)', #-
                                             'Average charge voltage (first 15% of capacity) - 1st cycle',
                                            #  'Avg. of Real (Z) (10 cycles)', #-
                                            #  'Avg. of Img (Z) (10 cycles)', #-
                                             'Average charge voltage (first 5% of capacity) - 1st cycle', #-
                                             'Average charge voltage (first 10% of capacity) - 1st cycle',
                                             'Average charge voltage (between 5% &15% SoC) - 1st cycle',
                                             'Average charge voltage (between 10% &15% SoC) - 1st cycle',
                                             'Average charge voltage (between 5% &10% SoC) - 1st cycle',
                                             'Average discharge voltage (first 15% of capacity) - 1st cycle',
                                             'Average discharge voltage (first 10% of capacity) - 1st cycle',
                                             'Average discharge voltage (first 5% of capacity) - 1st cycle', #-
                                             'Average discharge voltage (between 5% &15% SoC) - 1st cycle',
                                             'Average discharge voltage (between 10% &15% SoC) - 1st cycle',
                                             'Average discharge voltage (between 5% &10% SoC) - 1st cycle'
                                             ]
                                             , cell)
        Xs = np.append(Xs, features)
        cell_life_cycle = df_label.loc[df_label['battery']==cell]['Corrected cycle life']
        Ys = np.append(Ys, cell_life_cycle)
    Xs = np.reshape(Xs, (len(cellsid), -1))
    print(Xs.shape)
    return Xs, Ys

# def get_Li_metal_all_feature_dataset(eof='all'):
#     cellsid = get_cell_id_from_failure_mode(eof=eof)
#     df_label = pd.read_csv('BatteryData/Li_metal_label.csv')
#     Xs = np.array([])
#     Ys = np.array([])
#     for cell in cellsid:
#         features = get_features_df(df_label,[
#                                             #  'Var of avg. disch. Voltage (10 cycles)', #-
#                                              'Average discharge voltage (1st cycle)',
#                                             #  'Average discharge voltage (10th cycle)', #-
#                                              'Average charge voltage (1st cycle)',
#                                             #  'Var of avg. chg. Voltage (10 cycles)', #-
#                                              'Average charge voltage (first 15% of capacity) - 1st cycle',
#                                              'Avg. of Real (Z) (10 cycles)', #-
#                                              'Avg. of Img (Z) (10 cycles)', #-
#                                             #  'Average charge voltage (first 5% of capacity) - 1st cycle', #-
#                                              'Average charge voltage (first 10% of capacity) - 1st cycle',
#                                              'Average charge voltage (between 5% &15% SoC) - 1st cycle',
#                                              'Average charge voltage (between 10% &15% SoC) - 1st cycle',
#                                             #  'Average charge voltage (between 5% &10% SoC) - 1st cycle',
#                                              'Average discharge voltage (first 15% of capacity) - 1st cycle',
#                                              'Average discharge voltage (first 10% of capacity) - 1st cycle',
#                                             #  'Average discharge voltage (first 5% of capacity) - 1st cycle', #-
#                                              'Average discharge voltage (between 5% &15% SoC) - 1st cycle',
#                                              'Average discharge voltage (between 10% &15% SoC) - 1st cycle',
#                                              'Average discharge voltage (between 5% &10% SoC) - 1st cycle'
#                                              ]
#                                              , cell)
#         cell_EIS = pd.read_csv('BatteryData/Cell '+str(cell) + ' - EIS.csv')
#         delta_rez, var_delta_rez, delta_imz, var_delta_imz = get_deltaEIS_feature(cell_EIS)
#         EIS_feature = get_EIS_feature(cell)
#         features.extend(EIS_feature)
#         # print(var_delta_rez)
#         features.append(var_delta_rez)
#         features.extend(delta_imz)
#         features.append(var_delta_imz)
#         features.extend(delta_rez)
#         Xs = np.append(Xs, features)
#         cell_life_cycle = df_label.loc[df_label['battery']==cell]['Corrected cycle life']
#         Ys = np.append(Ys, cell_life_cycle)
#     Xs = np.reshape(Xs, (len(cellsid), -1))
#     print(Xs.shape)
#     return Xs, Ys


def get_Li_metal_all_feature_dataset_short_circuit(eof='all'):
    cellsid = get_cell_id_from_failure_mode(eof=eof)
    df_label = pd.read_csv('BatteryData/Li_metal_label.csv')
    le = preprocessing.LabelEncoder()
    le.fit(df_label['Failure mode'])
    df_label['Failure mode label'] = le.transform(df_label['Failure mode'])
    Xs = np.array([])
    Ys = np.array([])
    for cell in cellsid:
        features = get_features_df(df_label,[
                                             'Var of avg. disch. Voltage (10 cycles)', #-
                                             'Average discharge voltage (1st cycle)',
                                             'Average discharge voltage (10th cycle)', #-
                                             'Average charge voltage (1st cycle)',
                                             'Var of avg. chg. Voltage (10 cycles)', #-
                                             'Average charge voltage (first 15% of capacity) - 1st cycle',
                                             'Avg. of Real (Z) (10 cycles)', #-
                                             'Avg. of Img (Z) (10 cycles)', #-
                                             'Average charge voltage (first 5% of capacity) - 1st cycle', #-
                                             'Average charge voltage (first 10% of capacity) - 1st cycle',
                                             'Average charge voltage (between 5% &15% SoC) - 1st cycle',
                                             'Average charge voltage (between 10% &15% SoC) - 1st cycle',
                                             'Average discharge voltage (first 15% of capacity) - 1st cycle',
                                             'Average discharge voltage (first 10% of capacity) - 1st cycle',
                                             'Average discharge voltage (first 5% of capacity) - 1st cycle', #-
                                             'Average discharge voltage (between 5% &15% SoC) - 1st cycle',
                                             'Average discharge voltage (between 10% &15% SoC) - 1st cycle',
                                             'Average discharge voltage (between 5% &10% SoC) - 1st cycle'
                                             ]
                                             , cell)
        cell_EIS = pd.read_csv('BatteryData/Cell '+str(cell) + ' - EIS.csv')
        delta_rez, var_delta_rez, delta_imz, var_delta_imz = get_deltaEIS_feature(cell_EIS)
        EIS_feature = get_EIS_feature(cell)
        features.extend(EIS_feature)
        # print(var_delta_rez)
        features.append(var_delta_rez)
        features.extend(delta_imz)
        features.append(var_delta_imz)
        features.extend(delta_rez)
        Xs = np.append(Xs, features)
        cell_life_cycle = df_label.loc[df_label['battery']==cell]['Failure mode label']
        Ys = np.append(Ys, cell_life_cycle)
    Xs = np.reshape(Xs, (len(cellsid), -1))
    print(Xs.shape)
    print(Ys)
    return Xs, Ys