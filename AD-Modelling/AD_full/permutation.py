


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
my_f = pd.read_csv(dpath + 'Results/DM_full/s03_DM_full.csv')['Features'].tolist()
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

my_gbm = LGBMClassifier(objective='binary',
                                    metric='auc',
                                    is_unbalance=True,
                                    n_jobs=4,
                                    verbosity=-1, seed=2020)
my_gbm.set_params(**my_params)

def permute_epoch(clf_fitted, X, y, raw_auc, n_repos):
    delta_lst = []
    i = 0
    while i < n_repos:
        delta = []
        for f in X.columns:
            X_permuted = X.copy()
            X_permuted[f] = X_permuted[f].sample(frac=1, replace=True, axis=0, ignore_index=True)
            y_pred_prob_permuted = clf_fitted.predict_proba(X_permuted)[:, 1]
            auc_permuted = roc_auc_score(y, y_pred_prob_permuted)
            delta.append(raw_auc - auc_permuted)
        i += 1
        delta_lst.append(delta)
    return pd.DataFrame(delta_lst).T


def permutation(mykf, clf, X, y, n_repos):
    delta_lst = []
    for train_idx, test_idx in mykf.split(X, y):
        X_train, y_train = X.iloc[train_idx,:], y.iloc[train_idx]
        X_test, y_test = X.iloc[test_idx,:], y.iloc[test_idx]
        clf.fit(X_train, y_train)
        y_pred_prob = clf.predict_proba(X_test)[:, 1]
        auc_orginal = roc_auc_score(y_test, y_pred_prob)
        delta_df = permute_epoch(clf, X_test, y_test, auc_orginal, n_repos = n_repos)
        delta_lst.append(delta_df)
    return delta_lst, X.columns

delta_lst, _ = permutation(mykf, my_gbm, X, y, n_repos = 3)

delta_df = pd.DataFrame(delta_lst).T
delta_df.shape
delta_df.mean(axis = 1)
