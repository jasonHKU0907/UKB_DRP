

import numpy as np
import pandas as pd
import pickle
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import accuracy_score, recall_score, roc_auc_score, average_precision_score
from Utility.Processing_Utilities import *
from Utility.Training_Utilities import *
from Utility.Evaluation_Utilities import *
from lightgbm import LGBMClassifier
from sklearn.calibration import CalibratedClassifierCV
from sklearn.calibration import calibration_curve
from matplotlib import pyplot
import warnings
warnings.filterwarnings(('ignore'))

dpath = '/Volumes/JasonWork/UKB/'
my_f = pd.read_csv(dpath + 'Results/Results_woHES/DM_5yrs/s03_DM_5yrs.csv')['Features'][:10]
mydf = pd.read_csv(dpath + 'Preprocessed_Data/Preprocessed_Data.csv')
X = mydf[my_f]
y = mydf['dementia_status']
y[mydf['dementia_years']>5] = 0
mykf = StratifiedKFold(n_splits = 5, random_state = 2022, shuffle = True)

best_params = {'n_estimators': 500,
             'max_depth': 15,
             'num_leaves': 10,
             'subsample': 0.7,
             'learning_rate': 0.01,
             'colsample_bytree': 0.7}

def extend(my_array, nb_points):
    if len(my_array) == nb_points:
        pass
    else:
        nb2impute = nb_points - len(my_array)
        impute_array = np.zeros(nb2impute)
        my_array = np.concatenate((impute_array, my_array), axis=0)
    return np.expand_dims(my_array, -1)


cutoff_list = [np.round(0.001+i*0.001, 3) for i in range(100)]
results_cv = []
obs_array, pred_array = np.zeros((10, 1)), np.zeros((10, 1))
y_test_lst, y_pred_prob_lst = [], []


for train_idx, test_idx in mykf.split(X, y):
    X_train, X_test = X.iloc[train_idx,:], X.iloc[test_idx,:]
    y_train, y_test = y.iloc[train_idx], y.iloc[test_idx]
    my_gbm = LGBMClassifier(objective = 'binary', is_unbalance = True, n_jobs = 4,
                            metric = 'auc', verbose = -1, seed = 2022)
    my_gbm.set_params(**best_params)
    calibrate = CalibratedClassifierCV(my_gbm, method='isotonic', cv=5)
    calibrate.fit(X_train, y_train)
    y_pred_prob = calibrate.predict_proba(X_test)[:, 1]
    obsf, predf = calibration_curve(y_test, y_pred_prob, n_bins=10, strategy = 'quantile')
    obs_array = np.concatenate((obs_array, extend(obsf, nb_points=10)), axis=1)
    pred_array = np.concatenate((pred_array, extend(predf, nb_points=10)), axis=1)
    results_cv.append(get_full_eval(y_test, y_pred_prob, cutoff_list))
    y_pred_prob_lst.append(np.array(y_pred_prob))
    y_test_lst.append(np.array(y_test))

y_pred_prob_df = pd.DataFrame(y_pred_prob_lst).T
y_test_df = pd.DataFrame(y_test_lst).T
y_pred_prob_df.to_csv(dpath + 'Results/Results_woHES/DM_5yrs/pred_prob_cv_df.csv')
y_test_df.to_csv(dpath + 'Results/Results_woHES/DM_5yrs/test_cv_df.csv')

finaloutput = avg_results(results_cv)
finaloutput.to_csv(dpath + 'Results/Results_woHES/DM_5yrs/s05_DM_5yrs.csv')
finaloutput.iloc[:30, :9]

obs_array = 1000*obs_array[:, 1:]
pred_array = 1000*pred_array[:, 1:]

obs_mean = np.round(np.mean(obs_array, axis = 1),2)
pred_mean = np.round(np.mean(pred_array, axis = 1),2)
#pred_mean[0] = 0.01
print(obs_mean)
print(pred_mean)
x = np.arange(len(obs_mean)) +1
width = 0.35  # the width of the bars

fig, ax = pyplot.subplots(figsize=(15, 5))
rects1 = ax.bar(x - width/2, obs_mean, width, label='observed')
rects2 = ax.bar(x + width/2, pred_mean, width, label='predicted')
ax.set_ylabel('Probability ' + r'($\perthousand$)')
#ax.set_title('Calibration')
ax.set_xticks(x)
#ax.set_xticklabels(labels)
ax.legend(fontsize=18)
autolabel(rects1, ax, x_move = -0.05)
autolabel(rects2, ax, x_move = 0.05)
fig.tight_layout()
pyplot.show()



