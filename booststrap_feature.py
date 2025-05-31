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
import matplotlib.pyplot as plt

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
eof = '80'
# Xs, Ys = get_Li_metal_all_feature_dataset(eof=eof)
Xs, Ys = get_Li_metal_all_feature_dataset(eof=eof)
num_sample = len(Xs)
# var_avg_discharge_v_10 = 0
# avg_discharge_v_1 = 0
# avg_discharge_v_10 = 0
# avg_charge_v_1 = 0
# var_avg_charge_v_10 = 0
np.random.seed(42)
coefs = np.empty([0,len(Xs[0])])
for i in range(1):
    # idx = np.random.choice(np.arange(num_sample), int(num_sample*0.5), replace=False)
    # sampled_Xs = Xs[idx]
    # sampled_Ys = Ys[idx]

    sampled_Xs = Xs
    sampled_Ys = Ys

    scaler = StandardScaler().fit(sampled_Xs)
    X_train_scaled = scaler.transform(sampled_Xs)
    enet=ElasticNet()
    enet.fit(X_train_scaled, sampled_Ys)
    coefs = np.append(coefs,[enet.coef_],axis=0)
    # print(coefs.shape)

# print(coefs)
# print(coefs.shape)
# print(np.average(coefs,axis=0))
# abscoefs = np.abs(np.average(coefs,axis=0))
# ind = np.argpartition(abscoefs, -10)[-10:]
# print(ind)

selected = np.count_nonzero(coefs, axis=0)
# print(selected/100)
selected = selected/100
print('Var of avg. disch. Voltage (10 cycles)')
print(selected[0])
print('Average discharge voltage (1st cycle)')
print(selected[1])
print('Average charge voltage (10th cycle)')
print(selected[2])
print('Average discharge voltage (10th cycle)')
print(selected[3])
print('Average charge voltage (1st cycle)')
print(selected[4])
print('Var of avg. chg. Voltage (10 cycles)')
print(selected[5])
print('Avg. of Real (Z) (10 cycles)')
print(selected[6])
print('Avg. of Img (Z) (10 cycles)')
print(selected[7])
print('Average charge voltage (first 5% of capacity) - 1st cycle')
print(selected[8])
print('Average charge voltage (first 10% of capacity) - 1st cycle')
print(selected[9])
print('Average charge voltage (first 15% of capacity) - 1st cycle')
print(selected[10])
print('Average charge voltage (between 5% &15% SoC) - 1st cycle')
print(selected[11])
print('Average charge voltage (between 10% &15% SoC) - 1st cycle')
print(selected[12])
print('Average charge voltage (between 5% &10% SoC) - 1st cycle')
print(selected[13])
print('Average discharge voltage (first 15% of capacity) - 1st cycle')
print(selected[14])
print('Average discharge voltage (first 10% of capacity) - 1st cycle')
print(selected[15])
print('Average discharge voltage (first 5% of capacity) - 1st cycle')
print(selected[16])
print('Average discharge voltage (between 5% &15% SoC) - 1st cycle')
print(selected[17])
print('Average discharge voltage (between 10% &15% SoC) - 1st cycle')
print(selected[18])
print('Average discharge voltage (between 5% &10% SoC) - 1st cycle')
print(selected[19])
print('EIS_feature')
print(selected[20:80])

EIS = selected[20:80]
freqset = np.linspace(1, 9990, num=30)
rez = np.array(EIS[::2])
imz = np.array(EIS[1::2])
plt.title("EIS RealZ Stability Selection")
plt.plot(freqset, rez, color="red")
plt.xlabel('Frequency (Hz)')
plt.ylabel('Probability')
plt.savefig('plt/EIS_RealZ_Stability_Selection_'+eof+'.png')
plt.close()

plt.title("EIS ImZ Stability Selection")
plt.plot(freqset, imz, color="red")
plt.xlabel('Frequency (Hz)')
plt.ylabel('Probability')
plt.savefig('plt/EIS_ImZ_Stability_Selection_'+eof+'.png')
plt.close()

print('var_delta_rez')
print(selected[80])
print('delta_imz')
print(selected[81:111])

deltaImz = selected[81:111]
plt.title("Delta ImZ (10 cycles) Stability Selection")
plt.plot(freqset, deltaImz, color="red")
plt.xlabel('Frequency (Hz)')
plt.ylabel('Probability')
plt.savefig('plt/EIS_Delta_ImZ_Stability_Selection_'+eof+'.png')
plt.close()

print('var_delta_imz')
print(selected[111])
print('delta_rez')
print(selected[112:142])

deltaRealz = selected[112:142]
plt.title("Delta RealZ (10 cycles) Stability Selection")
plt.plot(freqset, deltaRealz, color="red")
plt.xlabel('Frequency (Hz)')
plt.ylabel('Probability')
plt.savefig('plt/EIS_Delta_RealZ_Stability_Selection_'+eof+'.png')
plt.close()