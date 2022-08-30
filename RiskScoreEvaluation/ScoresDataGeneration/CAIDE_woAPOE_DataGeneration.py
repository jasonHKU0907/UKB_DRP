

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

mydf_age = mydf[['eid', 'Age']]
mydf_age['age_caide'] = mydf_age['Age']
mydf_age['age_caide'].loc[mydf_age['age_caide']<47] = 0
mydf_age['age_caide'].loc[(mydf_age['age_caide']>=47) & (mydf_age['age_caide']<=53)] = 3
mydf_age['age_caide'].loc[mydf_age['age_caide']>53] = 4
mydf_age['age_caide'].value_counts()


mydf_educ = pd.read_csv(dpath + 'Preprocessed_Data/ScoreFeatures/EducYears.csv', usecols=['eid', '845-0.0'])
mydf_educ['educ_caide'] = mydf_educ['845-0.0'] - 5
mydf_educ['educ_caide'].loc[mydf_educ['educ_caide']<=6] = 3
mydf_educ['educ_caide'].loc[(mydf_educ['educ_caide']>6) & (mydf_educ['educ_caide']<10)] = 2
mydf_educ['educ_caide'].loc[mydf_educ['educ_caide']>=10] = 0
mydf_educ['educ_caide'].value_counts()


mydf_sex = mydf[['eid', '31-0.0']]
mydf_sex['sex_caide'] = mydf_sex['31-0.0']
mydf_sex['sex_caide'].value_counts()

mydf_sbp = mydf[['eid', '4080-0.0', '4080-0.1']]
mydf_sbp['4080-0.0'].loc[mydf_sbp['4080-0.0'].isnull() == True] = mydf_sbp['4080-0.1'].loc[mydf_sbp['4080-0.0'].isnull() == True]
mydf_sbp['4080-0.1'].loc[mydf_sbp['4080-0.1'].isnull() == True] = mydf_sbp['4080-0.0'].loc[mydf_sbp['4080-0.1'].isnull() == True]
mydf_sbp['sbp'] = (mydf_sbp['4080-0.0'] + mydf_sbp['4080-0.1'])/2
mydf_sbp['sbp'].loc[mydf_sbp['sbp'].isnull() == True] = mydf_sbp['sbp'].median()
mydf_sbp['sbp_caide'] = mydf_sbp['sbp']
mydf_sbp['sbp_caide'].loc[mydf_sbp['sbp_caide']<140] = 0
mydf_sbp['sbp_caide'].loc[mydf_sbp['sbp_caide']>=140] = 2
mydf_sbp['sbp_caide'].value_counts()


mydf_bmi = mydf[['eid', '21001-0.0', '23104-0.0']]
mydf_bmi['21001-0.0'].loc[mydf_bmi['21001-0.0'].isnull() == True] = mydf_bmi['23104-0.0'].loc[mydf_bmi['21001-0.0'].isnull() == True]
mydf_bmi['23104-0.0'].loc[mydf_bmi['23104-0.0'].isnull() == True] = mydf_bmi['21001-0.0'].loc[mydf_bmi['23104-0.0'].isnull() == True]
mydf_bmi['bmi'] = (mydf_bmi['21001-0.0'] + mydf_bmi['23104-0.0'])/2
mydf_bmi['bmi'].loc[mydf_bmi['bmi'].isnull() == True] = mydf_bmi['bmi'].median()
mydf_bmi['bmi_caide'] = mydf_bmi['bmi']
mydf_bmi['bmi_caide'].loc[mydf_bmi['bmi_caide']<=30] = 0
mydf_bmi['bmi_caide'].loc[mydf_bmi['bmi_caide']>30] = 2
mydf_bmi['bmi_caide'].value_counts()


mydf_chol = mydf[['eid', '30690-0.0']]
mydf_chol['chol'] = mydf_chol['30690-0.0']
mydf_chol['chol'].loc[mydf_chol['chol'].isnull() == True] = mydf_chol['chol'].median()
mydf_chol['chol_caide'] = mydf_chol['chol']
mydf_chol['chol_caide'].loc[mydf_chol['chol_caide']<=6.5] = 0
mydf_chol['chol_caide'].loc[mydf_chol['chol_caide']>6.5] = 2
mydf_chol['chol_caide'].value_counts()


mydf_acti = pd.read_csv(dpath + 'Preprocessed_Data/ScoreFeatures/Activity.csv', usecols=['eid', '22032-0.0'])
mydf_acti['acti_caide'] = mydf_acti['22032-0.0']
mydf_acti['acti_caide'].loc[mydf_acti['acti_caide']<=.5] = -1
mydf_acti['acti_caide'].loc[mydf_acti['acti_caide']>.5] = 0
mydf_acti['acti_caide'].loc[mydf_acti['acti_caide']==-1] = 1
mydf_acti['acti_caide'].value_counts()

mydf_caide = pd.merge(mydf[['eid', 'dementia_status', 'dementia_years', 'AD_status', 'AD_years']],
                      mydf_age[['eid', 'age_caide']], how='inner', on=['eid'])
mydf_caide = pd.merge(mydf_caide, mydf_educ[['eid', 'educ_caide']], how='inner', on=['eid'])
mydf_caide = pd.merge(mydf_caide, mydf_sex[['eid', 'sex_caide']], how='inner', on=['eid'])
mydf_caide = pd.merge(mydf_caide, mydf_sbp[['eid', 'sbp_caide']], how='inner', on=['eid'])
mydf_caide = pd.merge(mydf_caide, mydf_bmi[['eid', 'bmi_caide']], how='inner', on=['eid'])
mydf_caide = pd.merge(mydf_caide, mydf_chol[['eid', 'chol_caide']], how='inner', on=['eid'])
mydf_caide = pd.merge(mydf_caide, mydf_acti[['eid', 'acti_caide']], how='inner', on=['eid'])


mydf_caide.to_csv(dpath + 'Preprocessed_Data/ScoreData/CAIDE_data_wo_APOE.csv')
print('finished')
