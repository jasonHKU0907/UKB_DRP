
import matplotlib.pyplot as plt
import numpy as np
from scipy import interp
from sklearn.metrics import roc_curve
import pandas as pd
from sklearn.model_selection import StratifiedKFold
from Utility.Training_Utilities import *
from Utility.Evaluation_Utilities import *

dpath = '/Volumes/JasonWork/UKB/'
mydf = pd.read_csv(dpath + 'Preprocessed_Data/ScoreData/DRS.csv')
#mydf1= pd.read_csv(dpath + 'Preprocessed_Data/Preprocessed_Data.csv')
#mydf['dementia_status'][mydf1['dementia_years']>10] = 0

mydf.columns
my_f = ['age', 'age_sq', 'gender', 'cal_yrs', 'depr_1', 'depr_2',
       'depr_3', 'depr_4', 'depr_5', 'bmi', 'bmi_sq', 'anti_ht', 'smok_never',
       'smok_past', 'smok_curr', 'age_sq.1', 'alcoh_prob', 'diabetes',
       'depres', 'stroke', 'af', 'asp_use']
X = mydf[my_f]
y = mydf['dementia_status']

mykf = StratifiedKFold(n_splits = 5, random_state = 2022, shuffle = True)
cutoff_list = [np.round(0.001+i*0.001, 3) for i in range(100)]
results_cv = []
y_test_lst, y_pred_prob_lst = [], []

for train_idx, test_idx in mykf.split(X, y):
    X_test, y_test = X.iloc[test_idx,:], y.iloc[test_idx]
    y_pred =  np.exp(
        0.209 * X_test['age'] - 0.003 * X_test['age_sq'] + 0.129 * X_test['gender'] + 0.045 * X_test['cal_yrs'] +
        0.013 * X_test['depr_2'] + 0.118 * X_test['depr_3'] + 0.202 * X_test['depr_4'] + 0.226 * X_test['depr_5'] -
        0.062 * X_test['bmi'] + 0.003 * X_test['bmi_sq'] - 0.132 * X_test['anti_ht'] - 0.068 * X_test['smok_past'] -
        0.087 * X_test['smok_curr'] + 0.444 * X_test['alcoh_prob'] + 0.287 * X_test['diabetes'] +
        0.834 * X_test['depres'] + 0.577 * X_test['stroke'] + 0.221 * X_test['af'] + 0.253 * X_test['asp_use'])
    y_pred_prob = y_pred / (1 + y_pred)
    y_pred_max = y_pred_prob.quantile(0.995)
    y_pred_prob  = (y_pred_max - y_pred_prob)/6
    y_pred_prob.iloc[np.where(y_pred_prob<0)] = 0
    results_cv.append(get_full_eval(y_test, y_pred_prob, cutoff_list))
    y_pred_prob_lst.append(np.array(y_pred_prob))
    y_test_lst.append(np.array(y_test))

y_pred_prob_df = pd.DataFrame(y_pred_prob_lst).T
y_test_df = pd.DataFrame(y_test_lst).T
y_pred_prob_df.to_csv(dpath + 'Results/Results_RiskScores/DM/DRS/pred_prob_cv_df.csv')
y_test_df.to_csv(dpath + 'Results/Results_RiskScores/DM/DRS/test_cv_df.csv')

finaloutput = avg_results(results_cv)
finaloutput.to_csv(dpath + 'Results/Results_RiskScores/DM/DRS/results_cv.csv')
finaloutput.iloc[:30, :9]

