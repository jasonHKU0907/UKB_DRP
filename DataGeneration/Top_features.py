
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


my_f1 = pd.read_csv(dpath + 'Results/Results_woHES/DM_full/s03_DM_full.csv')['Features'][:20].tolist()
my_f2 = pd.read_csv(dpath + 'Results/Results_woHES/DM_10yrs/s03_DM_10yrs.csv')['Features'][:20].tolist()
my_f3 = pd.read_csv(dpath + 'Results/Results_woHES/DM_5yrs/s03_DM_5yrs.csv')['Features'][:20].tolist()
my_f4 = pd.read_csv(dpath + 'Results/Results_AD_woHES/AD_full/s03_AD_full.csv')['Features'][:20].tolist()
my_f5 = pd.read_csv(dpath + 'Results/Results_AD_woHES/AD_5yrs/s03_AD_5yrs.csv')['Features'][:20].tolist()
my_f6 = pd.read_csv(dpath + 'Results/Results_AD_woHES/AD_10yrs/s03_AD_10yrs.csv')['Features'][:20].tolist()
my_f = list(set(my_f1 + my_f2 + my_f3 + my_f4 + my_f5 + my_f6))
my_f = my_f + ['eid', 'dementia_status', 'dementia_years', 'AD_status',
               'AD_years', 'VD_status', 'VD_years', 'stroke_status']
mydf = pd.read_csv(dpath + 'Preprocessed_Data/Preprocessed_Data.csv')
mydf = mydf[my_f]
mydf.to_csv(dpath + 'Preprocessed_Data/Preprocessed_Data_topf.csv')
