

import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedKFold
from Utility.Training_Utilities import *
from lightgbm import LGBMClassifier
import warnings
warnings.filterwarnings(('ignore'))

dpath = '/Volumes/JasonWork/UKB/'
mydf = pd.read_csv(dpath + 'Preprocessed_Data/Preprocessed_Data.csv')
mydf['AD_years'][mydf['AD_years']<0] = mydf['dementia_years'][mydf['AD_years']<0]
rm_f1 = ['Unnamed: 0', 'eid', 'dementia_status', 'dementia_years',
        'AD_status', 'AD_years', 'VD_status', 'VD_years', 'stroke_status']
rm_HES = ['41234-0.0', '41259-0.0', '41149-0.0', '41289-0.0',
          '41218-0.0', '41235-0.0', '41214-0.0', '41235-0.0']
rm_f = rm_f1 + rm_HES

X = mydf.drop(rm_f, axis = 1)
y = mydf['AD_status']
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

dict_f = pd.read_csv(dpath + 'Data/FieldID_selected.csv')
my_lst = pd.merge(my_imp_df, dict_f, how='inner', on=['Features'])
my_f = my_lst['Features']
my_pos_df = mydf.loc[mydf['AD_status'] == 1]
na_full = [round(mydf[ele].isnull().sum()*100/len(mydf),1) for ele in my_f]
na_pos = [round(my_pos_df[ele].isnull().sum()*100/len(my_pos_df),1) for ele in my_f]
myout = pd.concat((my_lst, pd.DataFrame(na_full), pd.DataFrame(na_pos)), axis=1)

myout.columns = ['Features', 'Cover', 'Gain', 'Path', 'Field', 'ValueType', 'Units', 'NA_full', 'NA_target']
myout.to_csv(dpath + 'Results/Results_AD_woHES/AD_full/s01_AD_full.csv')

print('finished')