

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
df_f = pd.read_csv(dpath + 'Results/Results_woHES/DM_5yrs/s03_DM_5yrs.csv')
#df_f.sort_values(by = 'Cover', ascending = False, inplace = True)
my_f = df_f['Features'].to_list()

mydf = pd.read_csv(dpath + 'Preprocessed_Data/Preprocessed_Data.csv')
X = mydf[my_f]
y = mydf['dementia_status']
y[mydf['dementia_years']>5] = 0

mykf = StratifiedKFold(n_splits = 5, random_state = 2022, shuffle = True)
my_params = {'n_estimators': 500,
             'max_depth': 15,
             'num_leaves': 10,
             'subsample': 0.7,
             'learning_rate': 0.01,
             'colsample_bytree': 0.7}

tmp_f, AUC_cv_lst= [], []
for f in my_f:
    tmp_f.append(f)
    my_X = X[tmp_f]
    AUC_cv = []
    for train_idx, test_idx in mykf.split(my_X, y):
        X_train, X_test = my_X.iloc[train_idx, :], my_X.iloc[test_idx, :]
        y_train, y_test = y.iloc[train_idx], y.iloc[test_idx]
        my_lgb = LGBMClassifier(objective='binary',
                                metric='auc',
                                is_unbalance=True,
                                n_jobs=4,
                                verbosity=-1, seed=2022)
        my_lgb.set_params(**my_params)
        my_lgb.fit(X_train, y_train)
        y_pred_prob = my_lgb.predict_proba(X_test)[:, 1]
        AUC_cv.append(roc_auc_score(y_test, y_pred_prob))
    tmp_out = np.array([np.mean(AUC_cv), np.std(AUC_cv)] + AUC_cv)
    AUC_cv_lst.append(np.round(tmp_out, 3))
    print((f, tmp_out))

AUC_df = pd.concat((pd.DataFrame({'Features':tmp_f}), pd.DataFrame(AUC_cv_lst)), axis = 1)
myout = pd.merge(AUC_df, dict_f, how='inner', on=['Features'])
myout.columns = ['Features', 'AUC_mean', 'AUC_std', 'AUC0', 'AUC1', 'AUC2', 'AUC3', 'AUC4',
                 'Path', 'Field', 'ValueType', 'Units']
myout = myout[['Features', 'AUC_mean', 'Path', 'Field', 'ValueType', 'Units',
               'AUC0', 'AUC1', 'AUC2', 'AUC3', 'AUC4','AUC_std']]

myout.to_csv(dpath + 'Results/Results_woHES/DM_5yrs/s041_DM_5yrs.csv')

