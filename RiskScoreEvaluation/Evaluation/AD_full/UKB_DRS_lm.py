


import matplotlib.pyplot as plt
import numpy as np
from scipy import interp
from sklearn.metrics import roc_curve
import pandas as pd
from sklearn.model_selection import StratifiedKFold
from Utility.Training_Utilities import *
from Utility.Evaluation_Utilities import *

dpath = '/Volumes/JasonWork/UKB/'
mydf = pd.read_csv(dpath + 'Preprocessed_Data/ScoreData/UKB_DRS.csv')
#mydf1= pd.read_csv(dpath + 'Preprocessed_Data/Preprocessed_Data.csv')
#mydf['dementia_status'][mydf1['dementia_years']>10] = 0

mydf.columns
my_f = ['Age', 'sex', 'educ_years', 'diabetes', 'depression',
       'stroke', 'fam_dem']
X = mydf[my_f]
y = mydf['AD_status']

mykf = StratifiedKFold(n_splits = 5, random_state = 2022, shuffle = True)
cutoff_list = [np.round(0.001+i*0.001, 3) for i in range(100)]
results_cv = []
y_test_lst, y_pred_prob_lst = [], []

for train_idx, test_idx in mykf.split(X, y):
    X_test, y_test = X.iloc[test_idx,:], y.iloc[test_idx]
    y_pred = np.exp(-13.529 + 0.144*X_test['Age'] + 0.284*X_test['sex'] - 0.054*X_test['educ_years'] +
                        0.504*X_test['diabetes'] + 0.791*X_test['depression'] + 0.900*X_test['stroke'] +
                        0.475*X_test['fam_dem'])
    y_pred_prob = y_pred/ (1+y_pred)
    results_cv.append(get_full_eval(y_test, y_pred_prob, cutoff_list))
    y_pred_prob_lst.append(np.array(y_pred_prob))
    y_test_lst.append(np.array(y_test))


y_pred_prob_df = pd.DataFrame(y_pred_prob_lst).T
y_test_df = pd.DataFrame(y_test_lst).T
y_pred_prob_df.to_csv(dpath + 'Results/Results_RiskScores/AD/UKB_DRS/pred_prob_cv_df.csv')
y_test_df.to_csv(dpath + 'Results/Results_RiskScores/AD/UKB_DRS/test_cv_df.csv')

finaloutput = avg_results(results_cv)
finaloutput.to_csv(dpath + 'Results/Results_RiskScores/AD/UKB_DRS/results_cv.csv')
finaloutput.iloc[:30, :9]

