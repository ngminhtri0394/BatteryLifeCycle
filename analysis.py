import os.path
import numpy as np
import matplotlib.pyplot as plt
import random

battery = random.choice(range(50))
print(battery)
file_name_list = [f'data/Data - 18-Nov-2022/Cell {battery} - Discharge.txt',
                  f'data/Data - 18-Nov-2022/Cell {battery} -Discharge.txt',
                  f'data/Data - 18-Nov-2022/Cell {battery}- Discharge.txt',
                  f'data/Data - 18-Nov-2022/Cell {battery}-Discharge.txt']

for i in file_name_list:
    if os.path.exists(i):
        file_name = i

with open(file_name) as file:
    lines = [line.strip().split('\t') for line in file]

cycle_list = [int(float(lines[i][0])) for i in range(1, len(lines))]
cycle_list = list(set(cycle_list))
print('Cycle num:', cycle_list[-1])

cycle_dict = {i: [[], []] for i in cycle_list}
for line in lines[1:]:
    cycle = int(float(line[0]))
    if cycle % 10 == 0:
        capacity = float(line[-1])
        ecell = float(line[-3])
        cycle_dict[cycle][0].append(capacity)
        cycle_dict[cycle][1].append(ecell)

for cycle in cycle_list:
    if cycle_dict[cycle][0]:
        plt.plot(cycle_dict[cycle][0], cycle_dict[cycle][1])
plt.title(f'Battery {battery}, {cycle_list[-1]} cycles')
plt.show()

