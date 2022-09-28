
import numpy as np
import pandas as pd
import sklearn
import scipy

dpath = '/Volumes/JasonWork/UKB/'

target_df = pd.read_csv(dpath + 'Preprocessed_Data/Dementia_target.csv')
mydata = pd.read_csv(dpath + 'Preprocessed_Data/Preprocessed_Data.csv')

mydata1 = pd.read_csv(dpath + 'Preprocessed_Data/ScoreFeatures/FamilyDementia.csv', usecols=['eid', 'fam_dem'])
mydata2 = pd.read_csv(dpath + 'Preprocessed_Data/ScoreFeatures/EducYears.csv', usecols=['eid', '845-0.0'])
mydata2['educ_yrs'] = mydata2['845-0.0'] - 5
mydata3 = pd.read_csv(dpath + 'Preprocessed_Data/ScoreFeatures/Diagnosed_Depression.csv', usecols=['eid', 'diag_depress'])
mydata4 = pd.read_csv(dpath + 'Preprocessed_Data/ScoreFeatures/Activity.csv', usecols=['eid', '22032-0.0'])
mydata4['activ'] = mydata4['22032-0.0']
mydata4['activ'].loc[mydata4['activ']<=.5] = 0
mydata4['activ'].loc[(mydata4['activ']>.5) & (mydata4['activ']<=1.5)] = 1
mydata4['activ'].loc[mydata4['activ']>1.5] = 2
mydata5 = pd.read_csv(dpath + 'Preprocessed_Data/ScoreFeatures/AlcoholProblem.csv', usecols=['eid', 'alcoh_prob'])
mydata6 = pd.read_csv(dpath + 'Preprocessed_Data/ScoreFeatures/Pesticides.csv', usecols=['eid', 'pesti'])
mydata7 = pd.read_csv(dpath + 'Preprocessed_Data/ScoreData/UKB_DRS.csv', usecols=['eid', 'diabetes'])
mydata8 = pd.read_csv(dpath + 'Preprocessed_Data/ScoreFeatures/NeverEatEggs.csv',
                      usecols=['eid', 'no_egg', 'no_dairy', 'no_wheat', 'no_sugar', 'eggdairywheatsugar'])
mydata9 = pd.read_csv(dpath + 'Preprocessed_Data/ScoreFeatures/DepressiveSymptoms.csv', usecols=['eid', 'depres_sym'])

mydf = pd.merge(mydata, mydata1, how='inner', on=['eid'])
mydf = pd.merge(mydf, mydata2[['eid', 'educ_yrs']], how='inner', on=['eid'])
mydf = pd.merge(mydf, mydata3, how='inner', on=['eid'])
mydf = pd.merge(mydf, mydata4, how='inner', on=['eid'])
mydf = pd.merge(mydf, mydata5, how='inner', on=['eid'])
mydf = pd.merge(mydf, mydata6, how='inner', on=['eid'])
mydf = pd.merge(mydf, mydata7, how='inner', on=['eid'])
mydf = pd.merge(mydf, mydata8, how='inner', on=['eid'])

rm_f = ['Unnamed: 0', '20008-0.0', '20010-0.0']
mydf.drop(rm_f, axis = 1, inplace = True)

mydf.to_csv(dpath + 'Preprocessed_Data/Preprocessed_Data.csv')

