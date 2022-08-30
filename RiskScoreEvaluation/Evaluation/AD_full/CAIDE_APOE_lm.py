


import matplotlib.pyplot as plt
import numpy as np
from scipy import interp
from sklearn.metrics import roc_curve
import pandas as pd
from sklearn.model_selection import StratifiedKFold
from Utility.Training_Utilities import *
from Utility.Evaluation_Utilities import *

dpath = '/Volumes/JasonWork/UKB/'
mydf = pd.read_csv(dpath + 'Preprocessed_Data/ScoreData/CAIDE_data_APOE.csv')
#mydf1= pd.read_csv(dpath + 'Preprocessed_Data/Preprocessed_Data.csv')
#mydf['dementia_status'][mydf1['dementia_years']>10] = 0

mydf.columns
mydf['Age47_53'] = mydf['age_caide']
mydf['Age47_53'].loc[mydf['Age47_53'] == 3] = 1.155
mydf['Age47_53'].loc[mydf['Age47_53'] == 5] = 0
mydf['Age53_'] = mydf['age_caide']
mydf['Age53_'].loc[mydf['Age53_'] == 3] = 0
mydf['Age53_'].loc[mydf['Age53_'] == 5] = 1.874

mydf['eduyrs7_9'] = mydf['educ_caide']
mydf['eduyrs7_9'].loc[mydf['eduyrs7_9'] == 3] = 1.149
mydf['eduyrs7_9'].loc[mydf['eduyrs7_9'] == 4] = 0
mydf['eduyrs_6'] = mydf['educ_caide']
mydf['eduyrs_6'].loc[mydf['eduyrs_6'] == 3] = 0
mydf['eduyrs_6'].loc[mydf['eduyrs_6'] == 4] = 1.587

mydf['sex'] = mydf['sex_caide']
mydf['sex'].loc[mydf['sex'] == 1] = 0.438

mydf['sbp'] = mydf['sbp_caide']
mydf['sbp'].loc[mydf['sbp'] == 2] = 0.817

mydf['bmi'] = mydf['bmi_caide']
mydf['bmi'].loc[mydf['bmi'] == 2] = 0.608

mydf['chol'] = mydf['chol_caide']
mydf['chol'].loc[mydf['chol'] == 1] = 0.460

mydf['acti'] = mydf['acti_caide']
mydf['acti'].loc[mydf['acti'] == 1] = 0.579

mydf['apoe'] = mydf['apoe_caide']
mydf['apoe'].loc[mydf['apoe'] == 2] = 0.890

my_f = ['Age47_53', 'Age53_', 'eduyrs7_9', 'eduyrs_6', 'sex', 'sbp', 'bmi', 'chol', 'acti', 'apoe']
X = mydf[my_f]
y = mydf['AD_status']

mykf = StratifiedKFold(n_splits = 5, random_state = 2022, shuffle = True)
cutoff_list = [np.round(0.001+i*0.001, 3) for i in range(100)]
results_cv = []
y_test_lst, y_pred_prob_lst = [], []

for train_idx, test_idx in mykf.split(X, y):
    X_test, y_test = X.iloc[test_idx,:], y.iloc[test_idx]
    y_pred = np.exp(-8.203 + X_test['Age47_53'] + X_test['Age53_'] + X_test['eduyrs7_9'] +
                        X_test['eduyrs_6'] + X_test['sex'] +  X_test['sbp'] + X_test['bmi'] +
                        X_test['chol'] + X_test['acti'] + X_test['apoe'])
    y_pred_prob = y_pred / (1 + y_pred)
    results_cv.append(get_full_eval(y_test, y_pred_prob, cutoff_list))
    y_pred_prob_lst.append(np.array(y_pred_prob))
    y_test_lst.append(np.array(y_test))

y_pred_prob_df = pd.DataFrame(y_pred_prob_lst).T
y_test_df = pd.DataFrame(y_test_lst).T
y_pred_prob_df.to_csv(dpath + 'Results/Results_RiskScores/AD/CAIDE_APOE/pred_prob_cv_df.csv')
y_test_df.to_csv(dpath + 'Results/Results_RiskScores/AD/CAIDE_APOE/test_cv_df.csv')

finaloutput = avg_results(results_cv)
finaloutput.to_csv(dpath + 'Results/Results_RiskScores/AD/CAIDE_APOE/results_cv.csv')
finaloutput.iloc[:30, :9]

