

import numpy as np
import pandas as pd
from Utility.Training_Utilities import *
from lightgbm import LGBMClassifier
from sklearn.model_selection import StratifiedKFold
import warnings
import shap
import matplotlib.pyplot as plt
warnings.filterwarnings(('ignore'))

dpath = '/Volumes/JasonWork/UKB/'
dict_f = pd.read_csv(dpath + 'Data/FieldID_selected1.csv')
my_f = pd.read_csv(dpath + 'Results/Results_AD_woHES/AD_5yrs/s041_AD_5yrs.csv')['Features'][:10]
mydf = pd.read_csv(dpath + 'Preprocessed_Data/Preprocessed_Data.csv')
mydf['AD_years'][mydf['AD_years']<0] = mydf['dementia_years'][mydf['AD_years']<0]
X = mydf[my_f]
y = mydf['AD_status']
y[mydf['AD_years']>5] = 0

my_label = match_labels(my_f, dict_f)
#my_label.iloc[2] = r'$APOE \epsilon 4$'
X.columns = my_label
mykf = StratifiedKFold(n_splits = 5, random_state = 2022, shuffle = True)

my_params = {'n_estimators': 500,
             'max_depth': 15,
             'num_leaves': 10,
             'subsample': 0.7,
             'learning_rate': 0.01,
             'colsample_bytree': 0.7}

for train_idx, test_idx in mykf.split(X, y):
    X_train, y_train = X.iloc[train_idx,:], y.iloc[train_idx]
    X_test, y_test = X.iloc[test_idx,:], y.iloc[test_idx]

my_lgb = LGBMClassifier(objective = 'binary',
                           metric = 'auc',
                           is_unbalance = True,
                           verbosity = 1, seed = 2020)
my_lgb.set_params(**my_params)
my_lgb.fit(X_train, y_train)

explainer = shap.Explainer(my_lgb)
shap_values = explainer(X_test)
#f_imp = np.sum(np.abs(shap_values[:, :, 1].values), axis = 0)
#my_order = np.argsort(f_imp)[::-1]
shap.plots.beeswarm(shap_values[:, :, 1], order=list(np.linspace(0, 9, 10).astype('uint8')))
plt.gcf().set_size_inches(18, 5.5)
ax = plt.gca()
ax.set_ylabel('Selected Predictors', fontsize = 20, weight = 'bold')
ax.set_xlabel('SHAP Values', fontsize = 16, weight = 'bold')
ylabels = [tick.get_text() for tick in ax.get_yticklabels()]
ax.set_yticklabels(ylabels, fontsize = 15, color = 'black')
plt.tight_layout()
plt.savefig(dpath + 'Results/Results_AD_woHES/AD_5yrs/Shap_Imp_5yrs.png')



fig = plt.figure(figsize = (8, 5))
plt.scatter(X_test['Age'], shap_values[:, 0, 1].base_values, marker = 's')
plt.show()

fig = plt.figure(figsize = (8, 5))
plt.scatter(X_test['APOE4'], shap_values[1][:, 2], marker = 's')
plt.show()

fig = plt.figure(figsize = (8, 5))
plt.scatter(X_test['400-0.1'], shap_values[1][:, 3], marker = 's')
plt.show()

fig = plt.figure(figsize = (8, 5))
plt.scatter(X_test['Time to complete round'], shap_values[:, 3, 1].values, marker = 's')
plt.show()

X_test.columns

fig = plt.figure(figsize = (8, 5))
plt.scatter(X_test['20023-0.0'], shap_values[1][:, 5], marker = 's')
plt.show()

fig = plt.figure(figsize = (8, 5))
plt.scatter(X_test['20010-0.0'], shap_values[1][:, 6], marker = 's')
plt.show()

fig = plt.figure(figsize = (8, 5))
plt.scatter(X_test['3526-0.0'], shap_values[1][:, 7], marker = 's')
plt.show()

fig = plt.figure(figsize = (8, 5))
plt.scatter(X_test['3064-0.2'], shap_values[1][:, 8], marker = 's')
plt.show()

fig = plt.figure(figsize = (8, 5))
plt.scatter(X_test['30710-0.0'], shap_values[1][:, 9], marker = 's')
plt.show()

