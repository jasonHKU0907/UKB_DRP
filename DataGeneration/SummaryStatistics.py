

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
my_f = ['eid', 'dementia_years', 'dementia_status', 'AD_years', 'AD_status',
        'Age', '31-0.0', 'educ_yrs', '21000-0.0', 'APOE4',
        'Time2CompeleteRoundTotal', '404-0.11',  '2188-0.0', '137-0.0',
        '23111-0.0', '3064-0.2', '3526-0.0', '30040-0.0']
mydf0 = pd.read_csv(dpath + 'Preprocessed_Data/Preprocessed_Data.csv', usecols = my_f)
mydf1 = pd.read_csv(dpath + 'Data/ukb45628.csv', usecols = ['eid', '21000-0.0', '845-0.0', '31-0.0'])
mydf = pd.merge(mydf0, mydf1, how = 'inner', on = ['eid'])
mydf['AD_years'][mydf['AD_years']<0] = mydf['dementia_years'][mydf['AD_years']<0]
X = mydf.copy()

X_dm = X.loc[X['dementia_status'] == 1]
X_nondm = X.loc[X['dementia_status'] == 0]
X_ad = X.loc[X['AD_status'] == 1]
X_nonad = X.loc[X['AD_status'] == 0]

df = X_ad.copy()
print((df['Age'].median(), df['Age'].quantile(0.25), df['Age'].quantile(0.75)))
print((df['31-0.0'].sum(), np.round(df['31-0.0'].sum()/len(df)*100,1)))
print((df['APOE4'].value_counts(), np.round(df['APOE4'].value_counts()/len(df),3)))
print(len(df) - df['APOE4'].value_counts().sum())
print((len(df) - df['APOE4'].value_counts().sum())/ len(df))

print((df['educ_yrs'].median(),
       df['educ_yrs'].quantile(0.25),
       df['educ_yrs'].quantile(0.75)))

df = X_nonad.copy()
print((df['845-0.0'].median(),
       df['845-0.0'].quantile(0.25),
       df['845-0.0'].quantile(0.75)))


(df['educ_yrs'].isnull() == True).sum()
np.round((df['educ_yrs'].isnull() == True).sum()/len(df)*100, 1)

X['21000-0.0_x'].value_counts()

print((df['Time2CompeleteRoundTotal'].median(),
       df['Time2CompeleteRoundTotal'].quantile(0.25),
       df['Time2CompeleteRoundTotal'].quantile(0.75)))

(df['Time2CompeleteRoundTotal'].isnull() == True).sum()
np.round((df['Time2CompeleteRoundTotal'].isnull() == True).sum()/len(df)*100, 1)


print((df['404-0.11'].median(),
       df['404-0.11'].quantile(0.25),
       df['404-0.11'].quantile(0.75)))

(df['404-0.11'].isnull() == True).sum()
np.round((df['404-0.11'].isnull() == True).sum()/len(df)*100, 1)



print((df['2188-0.0'].value_counts(), np.round(df['2188-0.0'].value_counts()/len(df),3)))

df['2188-0.0'].value_counts()
np.sum(df['2188-0.0'] == 1)/len(df)
len(df) - df['2188-0.0'].value_counts().sum() + np.sum(df['2188-0.0'] <0)
(len(df) - df['2188-0.0'].value_counts().sum() + np.sum(df['2188-0.0'] <0))/len(df)

print((df['137-0.0'].median(),
       df['137-0.0'].quantile(0.25),
       df['137-0.0'].quantile(0.75)))

(df['137-0.0'].isnull() == True).sum()
np.round((df['137-0.0'].isnull() == True).sum()/len(df)*100, 1)



print((df['23111-0.0'].median(),
       df['23111-0.0'].quantile(0.25),
       df['23111-0.0'].quantile(0.75)))

(df['23111-0.0'].isnull() == True).sum()
np.round((df['23111-0.0'].isnull() == True).sum()/len(df)*100, 1)

print((df['3064-0.2'].median(),
       df['3064-0.2'].quantile(0.25),
       df['3064-0.2'].quantile(0.75)))

(df['3064-0.2'].isnull() == True).sum()
np.round((df['3064-0.2'].isnull() == True).sum()/len(df)*100, 1)


print((df['3526-0.0'].median(),
       df['3526-0.0'].quantile(0.25),
       df['3526-0.0'].quantile(0.75)))

print((df['3526-0.0'].isnull() == True).sum())
np.round((df['3526-0.0'].isnull() == True).sum()/len(df)*100, 1)


print((df['30040-0.0'].median(),
       df['30040-0.0'].quantile(0.25),
       df['30040-0.0'].quantile(0.75)))

(df['30040-0.0'].isnull() == True).sum()
np.round((df['30040-0.0'].isnull() == True).sum()/len(df)*100, 1)






mydf['21000-0.0'].value_counts()
376107 + 13137 + 11159
400403 / 425159

mydf_dm = mydf.copy()
mydf_dm = mydf_dm.loc[mydf['dementia_status'] == 1]
mydf_dm['21000-0.0'].value_counts()
4724 + 173 + 110
5007/5287

mydf_nondm = mydf.copy()
mydf_nondm = mydf_nondm.loc[mydf['dementia_status'] == 0]
mydf_nondm['21000-0.0'].value_counts()
371383 + 13027 + 10986
395396 / 419872

mydf_ad = mydf.copy()
mydf_ad = mydf_ad.loc[mydf['AD_status'] == 1]
mydf_ad['21000-0.0'].value_counts()
2196 + 69 + 46
2311 / 2416



mydf_nonad = mydf.copy()
mydf_nonad = mydf_nonad.loc[mydf['AD_status'] == 0]
mydf_nonad['21000-0.0'].value_counts()
373911 + 13091 + 11090
398092 / mydf_nonad.shape[0]