


import matplotlib.pyplot as plt
import numpy as np
from scipy import interp
from sklearn.metrics import roc_curve
import pandas as pd
from sklearn.model_selection import StratifiedKFold
from Utility.Training_Utilities import *
from Utility.Evaluation_Utilities import *

dpath = '/Volumes/JasonWork/UKB/'
mydf = pd.read_csv(dpath + 'Preprocessed_Data/ScoreData/CAIDE_data_wo_APOE.csv')
#mydf1= pd.read_csv(dpath + 'Preprocessed_Data/Preprocessed_Data.csv')
#mydf['dementia_status'][mydf1['dementia_years']>10] = 0

mydf.columns
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

mykf = StratifiedKFold(n_splits = 5, random_state = 2022, shuffle = True)
cutoff_list = [np.round(0.001+i*0.001, 3) for i in range(100)]
results_cv = []
y_test_lst, y_pred_prob_lst = [], []

for train_idx, test_idx in mykf.split(X, y):
    X_test, y_test = X.iloc[test_idx,:], y.iloc[test_idx]
    y_pred = np.exp(-7.642 + X_test['Age47_53'] + X_test['Age53_'] + X_test['eduyrs7_9'] +
                        X_test['eduyrs_6'] + X_test['sex'] +  X_test['sbp'] + X_test['bmi'] +
                        X_test['chol'] + X_test['acti'])
    y_pred_prob = y_pred / (1 + y_pred)
    results_cv.append(get_full_eval(y_test, y_pred_prob, cutoff_list))
    y_pred_prob_lst.append(np.array(y_pred_prob))
    y_test_lst.append(np.array(y_test))

y_pred_prob_df = pd.DataFrame(y_pred_prob_lst).T
y_test_df = pd.DataFrame(y_test_lst).T
y_pred_prob_df.to_csv(dpath + 'Results/Results_RiskScores/DM/CAIDE_woAPOE/pred_prob_cv_df.csv')
y_test_df.to_csv(dpath + 'Results/Results_RiskScores/DM/CAIDE_woAPOE/test_cv_df.csv')

finaloutput = avg_results(results_cv)
finaloutput.to_csv(dpath + 'Results/Results_RiskScores/DM/CAIDE_woAPOE/results_cv.csv')
finaloutput.iloc[:30, :9]


