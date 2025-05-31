# from feature_extraction import *
from Li_metal_feature_extraction import *
from sklearn.model_selection import LeaveOneOut
from sklearn.pipeline import make_pipeline
from sklearn.linear_model import LinearRegression,ElasticNet
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import cross_validate
import matplotlib.pyplot as plt
from matplotlib.ticker import MaxNLocator
from matplotlib import cm
import matplotlib

# for end in [25, 30, 35, 40, 45, 50]:
err = []
err_mape = []
ubound = []
lbound = []
bu = 0
bl = 0
best_rmse = 1000
best_mape = 1000
eof = '80'
for lowerbound in range(1,9990,100):
    for upperbound in range(lowerbound+1,9990,100):
        print('Lower bound: ', lowerbound)
        print('Upper bound: ', upperbound)
        rmse_mean, rmse_std = [], []
        mape_mean, mape_std = [], []
        for end in [25]:
            Xs, Ys = get_Li_metal_EIS_dataset(eof=eof,freqlb=lowerbound, frequb=upperbound)
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
        err.append(-np.mean(rmse_mean))
        err_mape.append(-np.mean(mape_mean))
        if -np.mean(rmse_mean) < best_rmse:
            best_rmse = -np.mean(rmse_mean)
            best_mape = -np.mean(mape_mean)
            bu = upperbound
            bl = lowerbound
            print('best lb: ', bl)
            print('best ub: ', bu)
        ubound.append(upperbound)
        lbound.append(lowerbound)
        # print('Standard deviation of MAPE:', np.std(mape_mean))
print('best lb: ', bl)
print('best ub: ', bu)
print('Best RMSE: ', best_rmse)
print('Best MAPE: ', best_mape)
fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')

surf = ax.plot_trisurf(ubound, lbound, err_mape, cmap=cm.jet, linewidth=0)
# plot = plt.pcolormesh(ubound, lbound, err, cmap='RdBu', shading='flat')
fig.colorbar(surf)

ax.xaxis.set_major_locator(MaxNLocator(5))
ax.yaxis.set_major_locator(MaxNLocator(6))
ax.zaxis.set_major_locator(MaxNLocator(5))

fig.tight_layout()
ax.set_xlabel('Upper bound')
ax.set_ylabel('Lower bound')
ax.set_zlabel('MAPE')

plt.savefig('EIS_freq_lb_ub_'+eof+'_MAPE.png') # or: