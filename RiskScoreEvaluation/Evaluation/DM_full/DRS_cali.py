

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
mydf = pd.read_csv(dpath + 'Preprocessed_Data/ScoreData/DRS.csv')

my_f = ['age', 'age_sq', 'gender', 'cal_yrs', 'depr_1', 'depr_2',
       'depr_3', 'depr_4', 'depr_5', 'bmi', 'bmi_sq', 'anti_ht', 'smok_never',
       'smok_past', 'smok_curr', 'age_sq.1', 'alcoh_prob', 'diabetes',
       'depres', 'stroke', 'af', 'asp_use']
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
    X_to_calibrate = np.exp(
        0.209 * X_train_cali['age'] - 0.003 * X_train_cali['age_sq'] + 0.129 * X_train_cali['gender'] + 0.045 * X_train_cali['cal_yrs'] +
        0.013 * X_train_cali['depr_2'] + 0.118 * X_train_cali['depr_3'] + 0.202 * X_train_cali['depr_4'] + 0.226 * X_train_cali['depr_5'] -
        0.062 * X_train_cali['bmi'] + 0.003 * X_train_cali['bmi_sq'] - 0.132 * X_train_cali['anti_ht'] - 0.068 * X_train_cali['smok_past'] -
        0.087 * X_train_cali['smok_curr'] + 0.444 * X_train_cali['alcoh_prob'] + 0.287 * X_train_cali['diabetes'] +
        0.834 * X_train_cali['depres'] + 0.577 * X_train_cali['stroke'] + 0.221 * X_train_cali['af'] + 0.253 * X_train_cali['asp_use'])
    X_to_calibrate = X_to_calibrate / (1 + X_to_calibrate)
    X_to_calibrate_max = X_to_calibrate.quantile(0.995)
    X_to_calibrate = (X_to_calibrate_max - X_to_calibrate) / 6
    X_to_calibrate.iloc[np.where(X_to_calibrate < 0)] = 0
    iso_reg = IsotonicRegression().fit(X_to_calibrate, y_deploy_train_cali)
    y_pred_deploy_mod_test = np.exp(
        0.209 * X_test['age'] - 0.003 * X_test['age_sq'] + 0.129 * X_test['gender'] + 0.045 * X_test['cal_yrs'] +
        0.013 * X_test['depr_2'] + 0.118 * X_test['depr_3'] + 0.202 * X_test['depr_4'] + 0.226 * X_test['depr_5'] -
        0.062 * X_test['bmi'] + 0.003 * X_test['bmi_sq'] - 0.132 * X_test['anti_ht'] - 0.068 * X_test['smok_past'] -
        0.087 * X_test['smok_curr'] + 0.444 * X_test['alcoh_prob'] + 0.287 * X_test['diabetes'] +
        0.834 * X_test['depres'] + 0.577 * X_test['stroke'] + 0.221 * X_test['af'] + 0.253 * X_test['asp_use'])
    y_pred_deploy_mod_test = y_pred_deploy_mod_test / (1 + y_pred_deploy_mod_test)
    y_pred_deploy_mod_test_max = y_pred_deploy_mod_test.quantile(0.995)
    y_pred_deploy_mod_test = (y_pred_deploy_mod_test_max - y_pred_deploy_mod_test) / 6
    y_pred_deploy_mod_test.iloc[np.where(y_pred_deploy_mod_test < 0)] = 0
    y_pred_deploy_mod_cali_test = iso_reg.predict(y_pred_deploy_mod_test)
    y_pred_deploy_mod_cali_test = pd.DataFrame(y_pred_deploy_mod_cali_test)
    y_pred_deploy_mod_cali_test.fillna(0, inplace=True)
    y_pred_deploy_mod_cali_test[y_pred_deploy_mod_cali_test<0] = 0
    y_pred_deploy_mod_cali_test = y_pred_deploy_mod_cali_test.iloc[:,0]
    try:
        obsf, predf = calibration_curve(y_deploy_test, y_pred_deploy_mod_cali_test, n_bins=10, strategy='quantile')
        obs_array = np.concatenate((obs_array, extend(obsf, nb_points=10)), axis=1)
        pred_array = np.concatenate((pred_array, extend(predf, nb_points=10)), axis=1)
        results_cv.append(get_full_eval(y_deploy_test, y_pred_deploy_mod_cali_test, cutoff_list))
        y_pred_prob_lst.append(np.array(y_pred_deploy_mod_cali_test))
        y_test_lst.append(np.array(y_deploy_test))
        i += 1
    except:
        pass



y_pred_prob_df = pd.DataFrame(y_pred_prob_lst).T
y_test_df = pd.DataFrame(y_test_lst).T
y_pred_prob_df.to_csv(dpath + 'Results/Results_RiskScores/DM/DRS/pred_prob_cv_cali_df.csv')
y_test_df.to_csv(dpath + 'Results/Results_RiskScores/DM/DRS/test_cv_cali_df.csv')

finaloutput = avg_results(results_cv)
finaloutput.to_csv(dpath + 'Results/Results_RiskScores/DM/DRS/results_cali_cv.csv')
finaloutput.iloc[:30, :9]

