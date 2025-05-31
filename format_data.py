import csv
import os.path

import pandas as pd
import glob

# df = pd.read_fwf('BatteryData/Cell 1 - Charge.txt')
# df.to_csv('BatteryData/Cell 1 - Charge.csv')

# with open("BatteryData/Cell 1 - Charge.txt") as fp:
#     for line in fp:
#         print(repr(line))

def standard_file():
    for cell in range(1, 50):
        if os.path.isfile('BatteryData/Cell ' + str(cell) + '-Discharge.txt'):
            filename='Cell ' + str(cell) + '-Discharge.txt'
        elif os.path.isfile('BatteryData/Cell ' + str(cell) + ' -Discharge.txt'):
            filename = 'Cell ' + str(cell) + ' -Discharge.txt'
        elif os.path.isfile('BatteryData/Cell ' + str(cell) + ' - Discharge.txt'):
            filename = 'Cell ' + str(cell) + ' - Discharge.txt'
        elif os.path.isfile('BatteryData/Cell ' + str(cell) + '- Discharge.txt'):
            filename = 'Cell ' + str(cell) + '- Discharge.txt'
        else:
            print(cell)
            continue
        df = pd.read_csv('BatteryData/'+filename, sep='\t', lineterminator='\n')
        df.drop(df.columns[len(df.columns) - 1], axis=1, inplace=True)
        newf='BatteryData/Cell ' + str(cell) + ' - Discharge'
        df.to_csv(newf+'.csv',index=False)

def standard_file_charge():
    for cell in range(1, 50):
        if os.path.isfile('BatteryData/Cell ' + str(cell) + '-Charge.txt'):
            filename='Cell ' + str(cell) + '-Charge.txt'
        elif os.path.isfile('BatteryData/Cell ' + str(cell) + ' -Charge.txt'):
            filename = 'Cell ' + str(cell) + ' -Charge.txt'
        elif os.path.isfile('BatteryData/Cell ' + str(cell) + ' - Charge.txt'):
            filename = 'Cell ' + str(cell) + ' - Charge.txt'
        elif os.path.isfile('BatteryData/Cell ' + str(cell) + '- Charge.txt'):
            filename = 'Cell ' + str(cell) + '- Charge.txt'
        else:
            print(cell)
            continue
        df = pd.read_csv('BatteryData/'+filename, sep='\t', lineterminator='\n')
        df.drop(df.columns[len(df.columns) - 1], axis=1, inplace=True)
        newf='BatteryData/Cell ' + str(cell) + ' - Charge'
        df.to_csv(newf+'.csv',index=False)

def standard_file_energy():
    for cell in range(1, 50):
        filename='Cell ' + str(cell) + ' - Energy - Charge.txt'

        # if os.path.isfile('BatteryData/Cell ' + str(cell) + ' - Energy - Charge.txt'):
        #     filename='Cell ' + str(cell) + ' - Energy - Discharge.txt'
        # else:
        #     filename='Cell ' + str(cell) + ' - Energy - Disharge.txt'

        df = pd.read_csv('BatteryData/'+filename, sep='\t', lineterminator='\n')
        df.drop(df.columns[len(df.columns) - 1], axis=1, inplace=True)
        newf='BatteryData/Cell ' + str(cell) + ' - Energy - Charge'
        df.to_csv(newf+'.csv',index=False)

standard_file_energy()
