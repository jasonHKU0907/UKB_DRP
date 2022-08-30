

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
from matplotlib import pyplot
import warnings
warnings.filterwarnings(('ignore'))

dpath = '/Volumes/JasonWork/UKB/'
mydf = pd.read_csv(dpath + 'Preprocessed_Data/Preprocessed_Data.csv')

mydf_age = mydf[['eid', 'Age', '31-0.0']]
mydf_age['age_male'] = mydf_age['Age']
mydf_age['age_male'].loc[mydf_age['31-0.0'] == 0] = 0
mydf_age['age_male'].loc[(mydf_age['age_male']>0) & (mydf_age['age_male']<65)] = 0
mydf_age['age_male'].loc[(mydf_age['age_male']>=65) & (mydf_age['age_male']<70)] = 1
mydf_age['age_male'].loc[(mydf_age['age_male']>=70) & (mydf_age['age_male']<75)] = 12
mydf_age['age_male'].loc[(mydf_age['age_male']>=75) & (mydf_age['age_male']<80)] = 18
mydf_age['age_male'].loc[(mydf_age['age_male']>=80) & (mydf_age['age_male']<85)] = 26
mydf_age['age_male'].loc[(mydf_age['age_male']>=85) & (mydf_age['age_male']<90)] = 33
mydf_age['age_male'].loc[mydf_age['age_male']>=90] = 38
mydf_age['age_male'].value_counts()

mydf_age['age_female'] = mydf_age['Age']
mydf_age['age_female'].loc[mydf_age['31-0.0'] == 1] = 0
mydf_age['age_female'].loc[(mydf_age['age_female']>0) & (mydf_age['age_female']<65)] = 0
mydf_age['age_female'].loc[(mydf_age['age_female']>=65) & (mydf_age['age_female']<70)] = 5
mydf_age['age_female'].loc[(mydf_age['age_female']>=70) & (mydf_age['age_female']<75)] = 14
mydf_age['age_female'].loc[(mydf_age['age_female']>=75) & (mydf_age['age_female']<80)] = 21
mydf_age['age_female'].loc[(mydf_age['age_female']>=80) & (mydf_age['age_female']<85)] = 29
mydf_age['age_female'].loc[(mydf_age['age_female']>=85) & (mydf_age['age_female']<90)] = 35
mydf_age['age_female'].loc[mydf_age['age_female']>=90] = 41
mydf_age['age_female'].value_counts()
mydf_age['age_full'] = mydf_age['age_male'] + mydf_age['age_female']
mydf_age['age_full'].value_counts()

mydf_educ = pd.read_csv(dpath + 'Preprocessed_Data/ScoreFeatures/EducYears.csv', usecols=['eid', '845-0.0'])
mydf_educ['educ_years'] = mydf_educ['845-0.0'] - 5
mydf_educ['educ_years'].loc[mydf_educ['educ_years']<=8] = 6
mydf_educ['educ_years'].loc[(mydf_educ['educ_years']>8) & (mydf_educ['educ_years']<=11)] = 3
mydf_educ['educ_years'].loc[mydf_educ['educ_years']>11] = 0
mydf_educ['educ_years'].value_counts()

mydf_bmi = mydf[['eid', '21001-0.0', '23104-0.0', 'Age']]
mydf_bmi['21001-0.0'].loc[mydf_bmi['21001-0.0'].isnull() == True] = mydf_bmi['23104-0.0'].loc[mydf_bmi['21001-0.0'].isnull() == True]
mydf_bmi['23104-0.0'].loc[mydf_bmi['23104-0.0'].isnull() == True] = mydf_bmi['21001-0.0'].loc[mydf_bmi['23104-0.0'].isnull() == True]
mydf_bmi['bmi'] = (mydf_bmi['21001-0.0'] + mydf_bmi['23104-0.0'])/2
mydf_bmi['bmi'].loc[mydf_bmi['bmi'].isnull() == True] = mydf_bmi['bmi'].median()
mydf_bmi['bmi'].loc[mydf_bmi['Age']>=60] = 0
mydf_bmi['bmi'].loc[mydf_bmi['bmi']<25] = 0
mydf_bmi['bmi'].loc[(mydf_bmi['bmi']>=25)&(mydf_bmi['bmi']<30)] = 2
mydf_bmi['bmi'].loc[mydf_bmi['bmi']>=30] = 5
mydf_bmi['bmi'].value_counts()

mydf_diab = mydf[['eid', '2443-0.0']]
mydf_diab['2443-0.0'].loc[mydf_diab['2443-0.0']<0] = 0
mydf_diab['diabetes'] = mydf_diab['2443-0.0']
mydf_diab['diabetes'].fillna(0, inplace = True)
mydf_diab['diabetes'].loc[mydf_diab['diabetes'] == 1] = 3
mydf_diab['diabetes'].value_counts()

mydf_depre = pd.read_csv(dpath + 'Preprocessed_Data/ScoreFeatures/Depression.csv', usecols=['eid', 'depres'])
mydf_depre['depression'] = mydf_depre['depres']
mydf_depre['depression'].loc[mydf_depre['depression'] >= 1] = 2
mydf_depre['depression'].value_counts()

mydf_chol = mydf[['eid', '30690-0.0', 'Age']]
mydf_chol['chol'] = mydf_chol['30690-0.0']
mydf_chol['chol'].loc[mydf_chol['chol'].isnull() == True] = mydf_chol['chol'].median()
mydf_chol['chol'].loc[mydf_chol['Age']>=60] = 0
mydf_chol['chol'].loc[mydf_chol['chol']<=6.5] = 0
mydf_chol['chol'].loc[mydf_chol['chol']>6.5] = 3
mydf_chol['chol'].value_counts()

mydf_smok = mydf[['eid', '20116-0.0']]
mydf_smok['smok'] = mydf_smok['20116-0.0']
mydf_smok['smok'].loc[mydf_smok['smok']<0] = np.nan
mydf_smok['smok'].loc[mydf_smok['smok'].isnull() == True] = 0
mydf_smok['smok'].loc[mydf_smok['smok']==2] = 4
mydf_smok['smok'].value_counts()

mydf_alcoh = mydf[['eid', '1558-0.0']]
mydf_alcoh['alcoh'] = mydf_alcoh['1558-0.0']
mydf_alcoh['alcoh'].loc[mydf_alcoh['alcoh']<0] = np.nan
mydf_alcoh['alcoh'].loc[mydf_alcoh['alcoh'].isnull() == True] = mydf_alcoh['alcoh'].median()
mydf_alcoh['alcoh'].loc[mydf_alcoh['alcoh']<=2] = 0
mydf_alcoh['alcoh'].loc[(mydf_alcoh['alcoh']>=3) & (mydf_alcoh['alcoh']<=4)] = -3
mydf_alcoh['alcoh'].loc[mydf_alcoh['alcoh']>=5] = 0
mydf_alcoh['alcoh'].value_counts()

mydf_social = mydf[['eid', '1031-0.0']]
mydf_social['social'] = mydf_social['1031-0.0']
mydf_social['social'].loc[mydf_social['social']<0] = np.nan
mydf_social['social'].loc[mydf_social['social'].isnull() == True] = mydf_social['social'].median()
mydf_social['social'].loc[mydf_social['social']==1] = 0
mydf_social['social'].loc[(mydf_social['social']>1) & (mydf_social['social']<=3)] = 1
mydf_social['social'].loc[(mydf_social['social']>3) & (mydf_social['social']<=5)] = 4
mydf_social['social'].loc[mydf_social['social']>5] = 6
mydf_social['social'].value_counts()

mydf_acti = pd.read_csv(dpath + 'Preprocessed_Data/ScoreFeatures/Activity.csv', usecols=['eid', '22032-0.0'])
mydf_acti['activ'] = mydf_acti['22032-0.0']
mydf_acti['activ'].loc[mydf_acti['activ']<=.5] = 0
mydf_acti['activ'].loc[(mydf_acti['activ']>.5) & (mydf_acti['activ']<=1.5)] = -2
mydf_acti['activ'].loc[mydf_acti['activ']>1.5] = -3
mydf_acti['activ'].value_counts()

mydf_cogni = pd.read_csv(dpath + 'Preprocessed_Data/Preprocessed_Data.csv', usecols=['eid', '20023-0.0'])
mydf_cogni['cogni'] = mydf_cogni['20023-0.0']
mydf_cogni['cogni'].loc[mydf_cogni['cogni'].isnull() == True] = mydf_cogni['cogni'].median()
mydf_cogni['cogni'].quantile(0.33)
#500
mydf_cogni['cogni'].quantile(0.67)
#582
mydf_cogni['cogni'].loc[mydf_cogni['cogni']<500] = -6
mydf_cogni['cogni'].loc[(mydf_cogni['cogni']>=500) & (mydf_cogni['cogni']<585)] = -7
mydf_cogni['cogni'].loc[mydf_cogni['cogni']>=585] = 0
mydf_cogni['cogni'].value_counts()

mydf_fish = mydf[['eid', '1329-0.0', '1339-0.0']]
mydf_fish['1329-0.0'].loc[mydf_fish['1329-0.0']<0] = np.nan
mydf_fish['1329-0.0'].loc[mydf_fish['1329-0.0'].isnull() == True] = mydf_fish['1329-0.0'].median()
mydf_fish['1339-0.0'].loc[mydf_fish['1339-0.0']<0] = np.nan
mydf_fish['1339-0.0'].loc[mydf_fish['1339-0.0'].isnull() == True] = mydf_fish['1339-0.0'].median()
mydf_fish['fish'] = pd.DataFrame(np.zeros((425159)))
mydf_fish['fish'].loc[(mydf_fish['1329-0.0']>=4) | (mydf_fish['1339-0.0']>=4) ] = -5
mydf_fish['fish'].loc[(mydf_fish['1329-0.0']==3) | (mydf_fish['1339-0.0']==3) & (mydf_fish['fish']>-1)] = -4
mydf_fish['fish'].loc[(mydf_fish['1329-0.0']==2) | (mydf_fish['1339-0.0']==2) & (mydf_fish['fish']>-1)] = -3
mydf_fish['fish'].loc[(mydf_fish['1329-0.0']<=1) | (mydf_fish['1339-0.0']<=1) & (mydf_fish['fish']>-1)] = 0
mydf_fish['fish'].value_counts()

mydf_pest = pd.read_csv(dpath + 'Preprocessed_Data/ScoreFeatures/Pesticides.csv', usecols=['eid', 'pesti'])
mydf_pest['pesti'].value_counts()


mydf_anuadri = pd.merge(mydf[['eid', 'dementia_status', 'dementia_years', 'AD_status', 'AD_years']],
                        mydf_age[['eid', 'age_full']], how='inner', on=['eid'])
mydf_anuadri = pd.merge(mydf_anuadri, mydf_educ[['eid', 'educ_years']], how='inner', on=['eid'])
mydf_anuadri = pd.merge(mydf_anuadri, mydf_bmi[['eid', 'bmi']], how='inner', on=['eid'])
mydf_anuadri = pd.merge(mydf_anuadri, mydf_diab[['eid', 'diabetes']], how='inner', on=['eid'])
mydf_anuadri = pd.merge(mydf_anuadri, mydf_depre[['eid', 'depression']], how='inner', on=['eid'])
mydf_anuadri = pd.merge(mydf_anuadri, mydf_chol[['eid', 'chol']], how='inner', on=['eid'])
mydf_anuadri = pd.merge(mydf_anuadri, mydf_smok[['eid', 'smok']], how='inner', on=['eid'])
mydf_anuadri = pd.merge(mydf_anuadri, mydf_alcoh[['eid', 'alcoh']], how='inner', on=['eid'])
mydf_anuadri = pd.merge(mydf_anuadri, mydf_social[['eid', 'social']], how='inner', on=['eid'])
mydf_anuadri = pd.merge(mydf_anuadri, mydf_acti[['eid', 'activ']], how='inner', on=['eid'])
mydf_anuadri = pd.merge(mydf_anuadri, mydf_cogni[['eid', 'cogni']], how='inner', on=['eid'])
mydf_anuadri = pd.merge(mydf_anuadri, mydf_fish[['eid', 'fish']], how='inner', on=['eid'])
mydf_anuadri = pd.merge(mydf_anuadri, mydf_pest[['eid', 'pesti']], how='inner', on=['eid'])

my_f = ['age_full', 'educ_years', 'bmi', 'diabetes', 'depression', 'chol',
        'smok', 'alcoh', 'social', 'activ', 'cogni', 'fish', 'pesti']
mydf_anuadri['anuadri_score'] = mydf_anuadri[my_f].sum(axis = 1)

mydf_anuadri.to_csv(dpath + 'Preprocessed_Data/ScoreData/ANU_ADRI.csv')

print('finished')