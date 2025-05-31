import pandas as pd
import numpy as np
from scipy.interpolate import splev, splrep
import pandas as pd
import matplotlib.pyplot as plt
import glob
import numpy as np
from scipy import interpolate

def get_energy_feature(id, numcycle):
    df = pd.read_csv('BatteryData/Cell ' + str(id) + ' - Energy - Discharge.csv')
    df['De'] = df.apply(lambda row: row['Energy discharge/W.h']/row['Q discharge/mA.h']*1000, axis=1)
    mincycle = df['cycle number'].min()
    maxcycle = df['cycle number'].max()
    cycle = []
    last_evs = []
    maxcaps = []
    for i in range(int(mincycle), int(mincycle)+numcycle):
        ev_i = df.loc[df['cycle number'] == i]
        maxcap = ev_i['Q discharge/mA.h'].values.tolist()[-1]
        last_ev_i = ev_i['De'].values.tolist()[-1]
        cycle.append(i)
        last_evs.append(last_ev_i)
        maxcaps.append(maxcap)
    # return pd.DataFrame({"cycle number": cycle, "last De": last_evs, "last Q": maxcaps})
    return np.array(last_evs)

def delta_e_eV_curve(df, i, j,vset):
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

def get_discharge_De_fade_curve_feature(df, start, end):
    df_cycle = df.groupby(['cycle number'], sort=False)['De'].max().reset_index()
    df_cycle = df_cycle[(df_cycle['cycle number'] >= start) & (df_cycle['cycle number'] <= end)]
    cap_curve = df_cycle['De'].to_list()
    log_cycle_number = np.log10(df_cycle['cycle number'].to_list())
    b = np.polyfit(log_cycle_number, cap_curve, deg=2)
    return np.array([b[0], b[1], df.loc[df['cycle number'] == start]['De'].min()])

def get_energy_eV_feature(id, numcycle):
    df = pd.read_csv('BatteryData/Cell ' + str(id) + ' - Energy - Discharge.csv')
    df['De'] = df.apply(lambda row: row['Energy discharge/W.h']/row['Q discharge/mA.h']*1000, axis=1)
    mincycle = df['cycle number'].min()
    maxcycle = df['cycle number'].max()
    cycle = []
    last_evs = []
    maxcaps = []
    for i in range(int(mincycle), int(mincycle)+numcycle):
        ev_i = df.loc[df['cycle number'] == i]
        maxcap = ev_i['Q discharge/mA.h'].values.tolist()[-1]
        last_ev_i = ev_i['De'].values.tolist()[-1]
        cycle.append(i)
        last_evs.append(last_ev_i)
        maxcaps.append(maxcap)
    # return pd.DataFrame({"cycle number": cycle, "last De": last_evs, "last Q": maxcaps})
    return np.array(last_evs)

def get_dataset_only_energy(num_cycle):
    df_label = pd.read_csv('BatteryData/clean_label.csv')
    Xs = np.array([])
    Ys = np.array([])
    cellsid = df_label['battery'].to_list()
    for cell in cellsid:
        df = pd.read_csv('BatteryData/Cell ' + str(cell) + ' - Energy - Discharge.csv')
        df['De'] = df.apply(lambda row: row['Energy discharge/W.h']/row['Q discharge/mA.h']*1000, axis=1)
        start = df['cycle number'].min()
        end = start + num_cycle
        De_feature = get_energy_feature(cell, num_cycle)
        fadecurve_feature = get_discharge_De_fade_curve_feature(df, start,end)
        feature = np.append(De_feature,fadecurve_feature)
        Xs = np.append(Xs,feature)
        Ys = np.append(Ys, df_label.loc[df_label['battery'] == cell]['life cycle'])
    return np.reshape(Xs,(len(cellsid),-1)), Ys

def plot_energy(id):
    df = pd.read_csv('BatteryData/Cell ' + str(id) + ' - Energy - Discharge.csv')
    i = df['cycle number'].min()+10
    j = i + 60
    df['eV'] = df.apply(lambda row: row['Energy discharge/W.h']/row['Q discharge/mA.h']*1000, axis=1)

    fig, ax = plt.subplots()
    eis_i = df.loc[df['cycle number'] == i]
    eis_j = df.loc[df['cycle number'] == j]
    eis_i.plot(y='eV', x='Ecell/V', kind='line', ax=ax, color='red')
    eis_j.plot(y='eV', x='Ecell/V', kind='line', ax=ax, color='blue')
    plt.show()

plot_energy(24)