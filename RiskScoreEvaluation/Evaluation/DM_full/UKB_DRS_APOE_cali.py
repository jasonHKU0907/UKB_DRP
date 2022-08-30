


import numpy as np
import pandas as pd
import pickle
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import accuracy_score, recall_score, roc_auc_score, average_precision_score
from Utility.Processing_Utilities import *
from Utility.Training_Utilities import *
from Utility.Evaluation_Utilities import *
from sklearn.calibration import CalibratedClassifierCV
from sklearn.calibration import calibration_curve
from sklearn.isotonic import IsotonicRegression

dpath = '/Volumes/JasonWork/UKB/'
mydf = pd.read_csv(dpath + 'Preprocessed_Data/ScoreData/UKB_DRS.csv')

my_f = ['Age', 'sex', 'educ_years', 'diabetes', 'depression',
       'stroke', 'fam_dem', 'apoe']
X = mydf[my_f]
y = mydf['dementia_status']
y_deploy = mydf['dementia_status'].copy()

mykf = StratifiedKFold(n_splits = 5, random_state = 2022, shuffle = True)
cutoff_list = [np.round(0.0001+i*0.0001, 5) for i in range(500)]
results_cv = []
obs_array, pred_array = np.zeros((10, 1)), np.zeros((10, 1))
y_test_lst, y_pred_prob_lst = [], []

i=0
for train_idx, test_idx in mykf.split(X, y):
    X_train, y_train, y_deploy_train = X.iloc[train_idx,:], y.iloc[train_idx], y_deploy.iloc[train_idx]
    X_test, y_deploy_test = X.iloc[test_idx,:], y_deploy.iloc[test_idx]
    X_train_mod, X_train_cali = X_train[68025:], X_train[:68025]
    y_train_mod, y_deploy_train_cali = y_train[68025:], y_deploy_train[:68025]
    X_to_calibrate = np.exp(-13.992 + 0.145*X_train_cali['Age'] + 0.275*X_train_cali['sex'] - 0.052*X_train_cali['educ_years'] +
                        0.529*X_train_cali['diabetes'] + 0.792*X_train_cali['depression'] + 0.904*X_train_cali['stroke'] +
                        0.362*X_train_cali['fam_dem'] + 0.960*X_train_cali['apoe'])
    iso_reg = IsotonicRegression().fit(X_to_calibrate, y_deploy_train_cali)
    y_pred_deploy_mod_test = np.exp(-13.992 + 0.145*X_test['Age'] + 0.275*X_test['sex'] - 0.052*X_test['educ_years'] +
                        0.529*X_test['diabetes'] + 0.792*X_test['depression'] + 0.904*X_test['stroke'] +
                        0.362*X_test['fam_dem'] + 0.960*X_test['apoe'])
    y_pred_deploy_mod_cali_test = iso_reg.predict(y_pred_deploy_mod_test)
    y_pred_deploy_mod_cali_test = pd.DataFrame(y_pred_deploy_mod_cali_test)
    y_pred_deploy_mod_cali_test.fillna(0, inplace=True)
    y_pred_deploy_mod_cali_test[y_pred_deploy_mod_cali_test<0] = 0
    y_pred_deploy_mod_cali_test = y_pred_deploy_mod_cali_test.iloc[:,0]
    obsf, predf = calibration_curve(y_deploy_test, y_pred_deploy_mod_cali_test, n_bins=10, strategy = 'quantile')
    obs_array = np.concatenate((obs_array, extend(obsf, nb_points=10)), axis=1)
    pred_array = np.concatenate((pred_array, extend(predf, nb_points=10)), axis=1)
    results_cv.append(get_full_eval(y_deploy_test, y_pred_deploy_mod_cali_test, cutoff_list))
    y_pred_prob_lst.append(np.array(y_pred_deploy_mod_cali_test))
    y_test_lst.append(np.array(y_deploy_test))
    i+=1



y_pred_prob_df = pd.DataFrame(y_pred_prob_lst).T
y_test_df = pd.DataFrame(y_test_lst).T
y_pred_prob_df.to_csv(dpath + 'Results/Results_RiskScores/DM/UKB_DRS_APOE/pred_prob_cv_cali_df.csv')
y_test_df.to_csv(dpath + 'Results/Results_RiskScores/DM/UKB_DRS_APOE/test_cv_cali_df.csv')

finaloutput = avg_results(results_cv)
finaloutput.to_csv(dpath + 'Results/Results_RiskScores/DM/UKB_DRS_APOE/results_cali_cv.csv')
finaloutput.iloc[:30, :9]


from matplotlib import pyplot
obs_array = 1000*obs_array[:, 1:]
pred_array = 1000*pred_array[:, 1:]

obs_mean = np.round(np.mean(obs_array, axis = 1),2)
pred_mean = np.round(np.mean(pred_array, axis = 1),2)
#pred_mean[0] = 0.05
print(obs_mean)
print(pred_mean)
x = np.arange(len(obs_mean)) +1
width = 0.35  #the width of the bars

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




