from Li_metal_feature_extraction import *
import numpy as np
import pandas as pd

Xs = np.array([])
df_label = pd.read_csv('BatteryData/Li_metal_label.csv')
cellsid = get_cell_id_from_failure_mode(eof=all)

features = ['Average charge voltage (1st cycle)',
'Average discharge voltage (1st cycle)',
'Average charge voltage (10th cycle)',
'Average discharge voltage (10th cycle)', 
'Var of avg. chg. Voltage (10 cycles)', 
'Var of avg. disch. Voltage (10 cycles)',
'Avg. of Real (Z) (10 cycles)', 
'Avg. of Img (Z) (10 cycles)', 
'Average charge voltage (first 5% of capacity) - 1st cycle', 
'Average discharge voltage (first 5% of capacity) - 1st cycle', 
'Average charge voltage (first 10% of capacity) - 1st cycle',
'Average discharge voltage (first 10% of capacity) - 1st cycle',
'Average charge voltage (first 15% of capacity) - 1st cycle',
'Average discharge voltage (first 15% of capacity) - 1st cycle',
'Average charge voltage (between 5% &15% SoC) - 1st cycle',
'Average discharge voltage (between 5% &15% SoC) - 1st cycle',
'Average charge voltage (between 10% &15% SoC) - 1st cycle',
'Average discharge voltage (between 10% &15% SoC) - 1st cycle',
'Average charge voltage (between 5% &10% SoC) - 1st cycle',
'Average discharge voltage (between 5% &10% SoC) - 1st cycle',
'Var delta Re Z',
'Var delta Im Z'
]
feat_dict = {}
for f1 in features:
    for cell in cellsid:
        if f1 == 'Var delta Re Z':
            cell_EIS = pd.read_csv('BatteryData/Cell '+str(cell) + ' - EIS.csv')
            delta_rez, var_delta_rez, delta_imz, var_delta_imz = get_deltaEIS_feature(cell_EIS)
            feature = var_delta_rez
        elif f1 == 'Var delta Im Z':
            cell_EIS = pd.read_csv('BatteryData/Cell '+str(cell) + ' - EIS.csv')
            delta_rez, var_delta_rez, delta_imz, var_delta_imz = get_deltaEIS_feature(cell_EIS)
            feature = var_delta_imz
        else:
            feature = get_features_df(df_label,[f1],cell)[0]
        if f1 not in feat_dict.keys():
            feat_dict[f1] = [feature]
        else:
            feat_dict[f1].append(feature)
all_coor = []
for f1 in feat_dict.keys():
    coor = []
    for f2 in feat_dict.keys():
        feat1 = feat_dict[f1]
        feat2 = feat_dict[f2]
        cov = np.corrcoef(feat1, feat2)
        coor.append(cov[0,1])
    all_coor.append(coor)

coordf = pd.DataFrame(all_coor, columns=features)    
coordf.to_csv('feature_coorelation.csv')
        
