


import matplotlib.pyplot as plt
import numpy as np
from scipy import interp
from sklearn.metrics import roc_curve
import pandas as pd

dpath = '/Volumes/JasonWork/UKB/'

my_f = ['eid', '31-0.0', '189-0.0', '21001-0.0', '23104-0.0',
        '20116-0.0', '2443-0.0', '131287-0.0', '131351-0.0']
mydf = pd.read_csv(dpath + 'Data/ukb45628.csv', usecols=my_f)
target_f = ['eid', 'Age', 'dementia_status', 'dementia_years', 'AD_status', 'AD_years']
target_df = pd.read_csv(dpath + 'Preprocessed_Data/Preprocessed_Data.csv', usecols=target_f)
mydf = pd.merge(mydf, target_df, how='inner', on=['eid'])

mydf['age'] = mydf['Age']
mydf['age_sq'] = mydf['Age']*mydf['Age']
mydf['gender'] = (mydf['31-0.0'] - 1)*(-1)
mydf['cal_yrs'] = pd.DataFrame(np.zeros(len(mydf)))
mydf['cal_yrs'].loc[mydf['cal_yrs'] == 0] = 12

mydf['deprivation'] = pd.DataFrame(np.zeros(len(mydf)))
mydf['189-0.0'].loc[mydf['189-0.0'].isnull() == True] = mydf['189-0.0'].median()
q2, q4, q6, q8 = mydf['189-0.0'].quantile([0.2, 0.4, 0.6, 0.8])
mydf['deprivation'].loc[mydf['189-0.0']<=q2] = 1
mydf['deprivation'].loc[(mydf['189-0.0']>q2) & (mydf['189-0.0']<=q4)] = 2
mydf['deprivation'].loc[(mydf['189-0.0']>q4) & (mydf['189-0.0']<=q6)] = 3
mydf['deprivation'].loc[(mydf['189-0.0']>q6) & (mydf['189-0.0']<=q8)] = 4
mydf['deprivation'].loc[mydf['189-0.0']>q8] = 5
mydf['depr_1'] = pd.DataFrame(np.zeros(len(mydf)))
mydf['depr_2'] = pd.DataFrame(np.zeros(len(mydf)))
mydf['depr_3'] = pd.DataFrame(np.zeros(len(mydf)))
mydf['depr_4'] = pd.DataFrame(np.zeros(len(mydf)))
mydf['depr_5'] = pd.DataFrame(np.zeros(len(mydf)))
mydf['depr_1'].loc[mydf['deprivation'] == 1] = 1
mydf['depr_2'].loc[mydf['deprivation'] == 2] = 1
mydf['depr_3'].loc[mydf['deprivation'] == 3] = 1
mydf['depr_4'].loc[mydf['deprivation'] == 4] = 1
mydf['depr_5'].loc[mydf['deprivation'] == 5] = 1
mydf['depr_5'].value_counts()

mydf_bmi = mydf[['eid', '21001-0.0', '23104-0.0']]
mydf_bmi['21001-0.0'].loc[mydf_bmi['21001-0.0'].isnull() == True] = mydf_bmi['23104-0.0'].loc[mydf_bmi['21001-0.0'].isnull() == True]
mydf_bmi['23104-0.0'].loc[mydf_bmi['23104-0.0'].isnull() == True] = mydf_bmi['21001-0.0'].loc[mydf_bmi['23104-0.0'].isnull() == True]
mydf_bmi['bmi'] = (mydf_bmi['21001-0.0'] + mydf_bmi['23104-0.0'])/2
mydf_bmi['bmi'].loc[mydf_bmi['bmi'].isnull() == True] = mydf_bmi['bmi'].median()
mydf['bmi'] = mydf_bmi['bmi']
mydf['bmi_sq'] = mydf['bmi']*mydf['bmi']

mydf['anti_ht'] = mydf['131287-0.0']
mydf['anti_ht'].fillna(0, inplace = True)
mydf['anti_ht'].loc[mydf['anti_ht']>0] = 1
mydf['anti_ht'].value_counts()

mydf_smok = mydf[['eid', '20116-0.0']]
mydf_smok['smok'] = mydf_smok['20116-0.0']
mydf_smok['smok'].loc[mydf_smok['smok']<0] = np.nan
mydf_smok['smok'].loc[mydf_smok['smok'].isnull() == True] = 0
mydf['smok'] = mydf_smok['smok']
mydf['smok'].value_counts()
mydf['smok_never'] = pd.DataFrame(np.zeros(len(mydf)))
mydf['smok_past'] = pd.DataFrame(np.zeros(len(mydf)))
mydf['smok_curr'] = pd.DataFrame(np.zeros(len(mydf)))
mydf['smok_never'].loc[mydf['smok'] == 0] = 1
mydf['smok_past'].loc[mydf['smok'] == 1] = 1
mydf['smok_curr'].loc[mydf['smok'] == 2] = 1
mydf['smok_curr'].value_counts()

mydf_alcoh = pd.read_csv(dpath + 'Preprocessed_Data/ScoreFeatures/AlcoholProblem.csv')
mydf['alcoh_prob'] = mydf_alcoh['alcoh_prob']
mydf['alcoh_prob'].value_counts()

mydf_diab = mydf[['eid', '2443-0.0']]
mydf_diab['2443-0.0'].loc[mydf_diab['2443-0.0']<0] = 0
mydf_diab['diabetes'] = mydf_diab['2443-0.0']
mydf_diab['diabetes'].fillna(0, inplace = True)
mydf['diabetes'] = mydf_diab['diabetes']
mydf['diabetes'].value_counts()


mydf_depre = pd.read_csv(dpath + 'Preprocessed_Data/ScoreFeatures/Depression.csv')
mydf['depres'] = mydf_depre['depres']
mydf['depres'].loc[mydf['depres']>0] = 1
mydf['depres'].value_counts()

mydf['stroke'] = pd.DataFrame(np.zeros(len(mydf)))

mydf['af'] = mydf['131287-0.0']
mydf['af'].fillna(0, inplace = True)
mydf['af'].loc[mydf['af']>0] = 1
mydf['af'].value_counts()

mydf_asp = pd.read_csv(dpath + 'Preprocessed_Data/ScoreFeatures/AspirinUse.csv')
mydf['asp_use'] = mydf_asp['asp_use']
mydf['asp_use'].value_counts()

my_f = ['eid', 'dementia_status', 'dementia_years', 'AD_status', 'AD_years',
        'age', 'age_sq', 'gender', 'cal_yrs',
        'depr_1', 'depr_2', 'depr_3', 'depr_4', 'depr_5', 'bmi', 'bmi_sq',
        'anti_ht', 'smok_never', 'smok_past', 'smok_curr', 'age_sq',
        'alcoh_prob', 'diabetes', 'depres', 'stroke', 'af', 'asp_use']

mydf_out = mydf[my_f]
mydf_out.to_csv(dpath + 'Preprocessed_Data/ScoreData/DRS.csv')
print('finished')
