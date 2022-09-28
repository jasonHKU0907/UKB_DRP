


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
import operator
import warnings
warnings.filterwarnings(('ignore'))

dpath = '/Volumes/JasonWork/UKB/'
dict_f = pd.read_csv(dpath + 'Data/FieldID_selected.csv')
my_f = pd.read_csv(dpath + 'Results_woHES/DM_full/s03_DM_full.csv')['Features'].tolist()
rm_envir = ['6146-0.0', '24503-0.0', '26410-0.0']
my_f = [f for f in my_f if f not in rm_envir]
mydf = pd.read_csv(dpath + 'Preprocessed_Data/Preprocessed_Data.csv')

X = mydf[my_f]
y = mydf['dementia_status']
mykf = StratifiedKFold(n_splits = 3, random_state = 2022, shuffle = True)

my_params = {'n_estimators': 500,
             'max_depth': 15,
             'num_leaves': 20,
             'subsample': 1,
             'learning_rate': 0.01,
             'colsample_bytree': 0.75}

def find_next_f(my_params, mykf, base_f, pool_f, X, y):
    AUC_list = []
    for f in pool_f:
        my_f = base_f + [f]
        my_X = X[my_f]
        AUC_cv = []
        for train_idx, test_idx in mykf.split(my_X, y):
            X_train, X_test = my_X.iloc[train_idx,:], my_X.iloc[test_idx,:]
            y_train, y_test = y.iloc[train_idx], y.iloc[test_idx]
            my_lgb = LGBMClassifier(objective='binary',
                                    metric='auc',
                                    is_unbalance=True,
                                    n_jobs=4,
                                    verbosity=-1, seed=2020)
            my_lgb.set_params(**my_params)
            my_lgb.fit(X_train, y_train)
            y_pred_prob = my_lgb.predict_proba(X_test)[:, 1]
            AUC_cv.append(roc_auc_score(y_test, y_pred_prob))
        AUC_list.append(np.round(np.average(AUC_cv), 4))
        print((np.round(np.average(AUC_cv), 4), f))
    index, best_auc = max(enumerate(AUC_list), key = operator.itemgetter(1))
    selected_f = pool_f[index]
    update_base = base_f + [selected_f]
    update_pool = pool_f
    update_pool.remove(selected_f)
    return ((best_auc, update_base, update_pool, selected_f))


base_f = ['Age', 'APOE4']
pool_f = list(X.columns)
pool_f = [f for f in pool_f if f not in base_f]
nb_f = len(pool_f)
my_auc, my_f = [0.787, 0.815], base_f


'''
base_f = pd.read_csv(dpath + 'Results/DM_full/s02_DM_full.csv')['Features'].tolist()
pool_f = list(X.columns)
pool_f = [f for f in pool_f if f not in base_f]
nb_f = len(pool_f)
my_auc = pd.read_csv(dpath + 'Results/DM_full/s02_DM_full.csv')['AUC'].tolist()
my_f = base_f
'''

for i in range(nb_f):
    my_params = my_params
    update_auc, update_base, update_pool, selected_f = find_next_f(my_params, mykf, base_f, pool_f, X, y)
    my_auc.append(update_auc)
    my_f.append(selected_f)
    base_f = update_base
    pool_f = update_pool
    my_auc_df = pd.DataFrame(my_auc, columns = ['AUC'])
    my_f_df = pd.DataFrame(my_f, columns = ['Features'])
    myout = pd.concat((my_auc_df, my_f_df), axis = 1)
    my_lst = pd.merge(myout, dict_f, how='inner', on=['Features'])
    my_f1 = my_lst['Features']
    my_pos_df = mydf.loc[mydf['dementia_status'] == 1]
    na_full = [round(mydf[ele].isnull().sum() * 100 / len(mydf), 1) for ele in my_f1]
    na_pos = [round(my_pos_df[ele].isnull().sum() * 100 / len(my_pos_df), 1) for ele in my_f1]
    finalout = pd.concat((my_lst, pd.DataFrame(na_full), pd.DataFrame(na_pos)), axis=1)
    finalout.columns = ['AUC', 'Features', 'Path', 'Field', 'ValueType', 'Units', 'NA_full', 'NA_target']
    finalout.to_csv(dpath + 'Results_woHES/DM_full/s040_DM_full.csv')




