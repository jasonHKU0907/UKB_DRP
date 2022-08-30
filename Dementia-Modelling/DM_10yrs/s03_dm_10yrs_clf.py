

import numpy as np
import pandas as pd
import pickle
from sklearn.model_selection import StratifiedKFold
from sklearn.feature_selection import RFECV
import matplotlib.pyplot as plt
from Utility.Evaluation_Utilities import *
from lightgbm import LGBMClassifier
from sklearn.calibration import CalibratedClassifierCV
from sklearn.calibration import calibration_curve
import operator
from scipy.stats import spearmanr
from scipy.cluster import hierarchy
from scipy.spatial.distance import squareform
from collections import defaultdict
from sklearn.inspection import permutation_importance

dpath = '/Volumes/JasonWork/UKB/'
dict_f = pd.read_csv(dpath + 'Data/FieldID_selected.csv')
my_f = pd.read_csv(dpath + 'Results/Results_woHES/DM_10yrs/s02_DM_10yrs.csv')['Features'].tolist()
mydf = pd.read_csv(dpath + 'Preprocessed_Data/Preprocessed_Data.csv')
X = mydf[my_f]
y = mydf['dementia_status']
y[mydf['dementia_years']>10] = 0
mykf = StratifiedKFold(n_splits = 5, random_state = 2022, shuffle = True)

my_params = {'n_estimators': 500,
             'max_depth': 15,
             'num_leaves': 10,
             'subsample': 0.7,
             'learning_rate': 0.01,
             'colsample_bytree': 0.7}

def normal_imp(mydict):
    mysum = sum(mydict.values())
    mykeys = mydict.keys()
    for key in mykeys:
        mydict[key] = mydict[key]/mysum
    return mydict

tg_imp_cv = Counter()
tc_imp_cv = Counter()

for train_idx, test_idx in mykf.split(X, y):
    X_train, y_train = X.iloc[train_idx,:], y.iloc[train_idx]
    my_lgb = LGBMClassifier(objective = 'binary',
                           metric = 'auc',
                           is_unbalance = True,
                           verbosity = 1, seed = 2020)
    my_lgb.set_params(**my_params)
    my_lgb.fit(X_train, y_train)
    totalgain_imp = my_lgb.booster_.feature_importance(importance_type='gain')
    totalgain_imp = dict(zip(my_lgb.booster_.feature_name(), totalgain_imp.tolist()))
    totalcover_imp = my_lgb.booster_.feature_importance(importance_type='split')
    totalcover_imp = dict(zip(my_lgb.booster_.feature_name(), totalcover_imp.tolist()))
    tg_imp_cv += Counter(normal_imp(totalgain_imp))
    tc_imp_cv += Counter(normal_imp(totalcover_imp))


tg_imp_df = pd.DataFrame({'Features': list(tg_imp_cv.keys()),
                          'TotalGain_cv': list(tg_imp_cv.values())})

tc_imp_df = pd.DataFrame({'Features': list(tc_imp_cv.keys()),
                          'TotalCover_cv': list(tc_imp_cv.values())})

my_imp_df = pd.merge(left = tc_imp_df, right = tg_imp_df, how = 'left')
my_imp_df.sort_values(by = 'TotalGain_cv', ascending = False, inplace = True)
my_imp_df['TotalGain_cv'] = my_imp_df['TotalGain_cv']/5
my_imp_df['TotalCover_cv'] = my_imp_df['TotalCover_cv']/5
dict_f = pd.read_csv(dpath + 'Data/FieldID_selected.csv')
my_lst = pd.merge(my_imp_df, dict_f, how='inner', on=['Features'])
my_f = my_lst['Features']
my_pos_df = mydf.loc[mydf['dementia_status'] == 1]
na_full = [round(mydf[ele].isnull().sum()*100/len(mydf),1) for ele in my_f]
na_pos = [round(my_pos_df[ele].isnull().sum()*100/len(my_pos_df),1) for ele in my_f]
myout = pd.concat((my_lst, pd.DataFrame(na_full), pd.DataFrame(na_pos)), axis=1)

myout.columns = ['Features', 'Cover', 'Gain', 'Path', 'Field', 'ValueType', 'Units', 'NA_full', 'NA_target']
myout.to_csv(dpath + 'Results/Results_woHES/DM_10yrs/s03_DM_10yrs.csv')

print('finished')