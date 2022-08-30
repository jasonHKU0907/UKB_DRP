


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
#mydf1= pd.read_csv(dpath + 'Preprocessed_Data/Preprocessed_Data.csv')
#mydf['dementia_status'][mydf1['dementia_years']>10] = 0

mydf.columns
my_f = ['Age', 'sex', 'educ_years', 'diabetes', 'depression',
       'stroke', 'fam_dem', 'apoe']
X = mydf[my_f]
y = mydf['AD_status']
y_deploy = mydf['AD_status'].copy()

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
    X_to_calibrate = np.exp(-13.529 + 0.144*X_train_cali['Age'] + 0.284*X_train_cali['sex'] - 0.054*X_train_cali['educ_years'] +
                        0.504*X_train_cali['diabetes'] + 0.791*X_train_cali['depression'] + 0.900*X_train_cali['stroke'] +
                        0.475*X_train_cali['fam_dem'])
    iso_reg = IsotonicRegression().fit(X_to_calibrate, y_deploy_train_cali)
    y_pred_deploy_mod_test = np.exp(-13.529 + 0.144*X_test['Age'] + 0.284*X_test['sex'] - 0.054*X_test['educ_years'] +
                        0.504*X_test['diabetes'] + 0.791*X_test['depression'] + 0.900*X_test['stroke'] +
                        0.475*X_test['fam_dem'])
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
y_pred_prob_df.to_csv(dpath + 'Results/Results_RiskScores/AD/UKB_DRS/pred_prob_cv_cali_df.csv')
y_test_df.to_csv(dpath + 'Results/Results_RiskScores/AD/UKB_DRS/test_cv_cali_df.csv')

finaloutput = avg_results(results_cv)
finaloutput.to_csv(dpath + 'Results/Results_RiskScores/AD/UKB_DRS/results_cali_cv.csv')
finaloutput.iloc[:30, :9]

