


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
mydf = pd.read_csv(dpath + 'Preprocessed_Data/ScoreData/CAIDE_data_wo_APOE.csv')

mydf['Age47_53'] = mydf['age_caide']
mydf['Age47_53'].loc[mydf['Age47_53'] == 3] = 1.084
mydf['Age47_53'].loc[mydf['Age47_53'] == 4] = 0
mydf['Age53_'] = mydf['age_caide']
mydf['Age53_'].loc[mydf['Age53_'] == 3] = 0
mydf['Age53_'].loc[mydf['Age53_'] == 4] = 1.762

mydf['eduyrs7_9'] = mydf['educ_caide']
mydf['eduyrs7_9'].loc[mydf['eduyrs7_9'] == 2] = 0.910
mydf['eduyrs7_9'].loc[mydf['eduyrs7_9'] == 3] = 0
mydf['eduyrs_6'] = mydf['educ_caide']
mydf['eduyrs_6'].loc[mydf['eduyrs_6'] == 2] = 0
mydf['eduyrs_6'].loc[mydf['eduyrs_6'] == 3] = 1.281

mydf['sex'] = mydf['sex_caide']
mydf['sex'].loc[mydf['sex'] == 1] = 0.470

mydf['sbp'] = mydf['sbp_caide']
mydf['sbp'].loc[mydf['sbp'] == 2] = 0.791

mydf['bmi'] = mydf['bmi_caide']
mydf['bmi'].loc[mydf['bmi'] == 2] = 0.831

mydf['chol'] = mydf['chol_caide']
mydf['chol'].loc[mydf['chol'] == 2] = 0.631

mydf['acti'] = mydf['acti_caide']
mydf['acti'].loc[mydf['acti'] == 1] = 0.527


my_f = ['Age47_53', 'Age53_', 'eduyrs7_9', 'eduyrs_6', 'sex', 'sbp', 'bmi', 'chol', 'acti']
X = mydf[my_f]
y = mydf['dementia_status']
y_deploy = mydf['dementia_status'].copy()

mykf = StratifiedKFold(n_splits = 5, random_state = 2022, shuffle = True)
cutoff_list = [np.round(0.0001+i*0.0001, 5) for i in range(1000)]
results_cv = []
obs_array, pred_array = np.zeros((10, 1)), np.zeros((10, 1))
y_test_lst, y_pred_prob_lst = [], []

i=0
for train_idx, test_idx in mykf.split(X, y):
    X_train, y_train, y_deploy_train = X.iloc[train_idx,:], y.iloc[train_idx], y_deploy.iloc[train_idx]
    X_test, y_deploy_test = X.iloc[test_idx,:], y_deploy.iloc[test_idx]
    X_train_mod, X_train_cali = X_train[168025:], X_train[:168025]
    y_train_mod, y_deploy_train_cali = y_train[168025:], y_deploy_train[:168025]
    X_to_calibrate = np.exp(-7.642 + X_train_cali['Age47_53'] + X_train_cali['Age53_'] + X_train_cali['eduyrs7_9'] +
                        X_train_cali['eduyrs_6'] + X_train_cali['sex'] +  X_train_cali['sbp'] + X_train_cali['bmi'] +
                        X_train_cali['chol'] + X_train_cali['acti'])
    iso_reg = IsotonicRegression().fit(X_to_calibrate, y_deploy_train_cali)
    y_pred_deploy_mod_test = np.exp(-7.642 + X_test['Age47_53'] + X_test['Age53_'] + X_test['eduyrs7_9'] +
                        X_test['eduyrs_6'] + X_test['sex'] +  X_test['sbp'] + X_test['bmi'] +
                        X_test['chol'] + X_test['acti'])
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
y_pred_prob_df.to_csv(dpath + 'Results/Results_RiskScores/DM/CAIDE_woAPOE/pred_prob_cv_cali_df.csv')
y_test_df.to_csv(dpath + 'Results/Results_RiskScores/DM/CAIDE_woAPOE/test_cv_cali_df.csv')

finaloutput = avg_results(results_cv)
finaloutput.to_csv(dpath + 'Results/Results_RiskScores/DM/CAIDE_woAPOE/results_cali_cv.csv')
finaloutput.iloc[:30, :9]


