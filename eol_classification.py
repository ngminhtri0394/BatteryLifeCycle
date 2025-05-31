from sklearn.model_selection import StratifiedKFold
from sklearn.linear_model import LogisticRegression
from Li_metal_feature_extraction import *
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
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
from sklearn.ensemble import RandomForestClassifier

Xs, Ys = get_Li_metal_all_feature_dataset_short_circuit('all')
skf = StratifiedKFold(n_splits=5)
model = RandomForestClassifier(n_estimators=15,max_depth=2, random_state=0)

acc_mean, acc_std = [], []
true_acc_mean, true_acc_std = [], []
auc_mean, auc_std = [], []
true_auc_mean, true_auc_std = [], []

pipe = make_pipeline(StandardScaler(), model)
scoring = {'auc': 'roc_auc',
           'accuracy_score': 'accuracy',}
        #    'true_mape': true_MAPE,
        #    'true_rmse': true_RMSE}
scores = cross_validate(pipe, Xs, Ys, cv=skf, scoring=scoring)

acc_mean.append(np.mean(scores['test_accuracy_score']))
acc_std.append(np.std(scores['test_accuracy_score']))

auc_mean.append(np.mean(scores['test_auc']))
auc_std.append(np.std(scores['test_auc']))


print('accuracy_score: %.3f (%.3f)' % (np.mean(acc_mean), np.mean(acc_std)))
print('auc: %.3f (%.3f)' % (np.mean(auc_mean), np.mean(auc_std)))