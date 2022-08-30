


import matplotlib.pyplot as plt
import numpy as np
from scipy import interp
from sklearn.metrics import roc_curve
import pandas as pd


dpath = '/Volumes/JasonWork/UKB/'

my_f = ['eid', 'APOE4', 'Age', '31-0.0', '2443-0.0', 'dementia_status', 'dementia_years', 'AD_status', 'AD_years']
mydf = pd.read_csv(dpath + 'Preprocessed_Data/preprocessed_data.csv', usecols = my_f)

mydf['sex'] = mydf['31-0.0']

mydf_educ = pd.read_csv(dpath + 'Preprocessed_Data/ScoreFeatures/EducYears.csv', usecols=['eid', '845-0.0'])
mydf_educ['educ_years'] = mydf_educ['845-0.0'] - 5
mydf_educ['educ_years'].loc[mydf_educ['educ_years']<=0] = 0
mydf_educ['educ_years'] = round(mydf_educ['educ_years'])
mydf_educ['educ_years'].value_counts()

mydf['2443-0.0'].loc[mydf['2443-0.0']<0] = 0
mydf['diabetes'] = mydf['2443-0.0']
mydf['diabetes'].fillna(0, inplace = True)
mydf['diabetes'].value_counts()

mydf_depre = pd.read_csv(dpath + 'Preprocessed_Data/ScoreFeatures/Depression.csv', usecols=['eid', 'depres'])
mydf_depre['depression'] = mydf_depre['depres']
mydf_depre['depression'].loc[mydf_depre['depression'] >= 1] = 1
mydf_depre['depression'].value_counts()

mydf['stroke'] = pd.DataFrame(np.zeros(len(mydf)))

mydf['apoe'] = mydf['APOE4']
mydf['apoe'].fillna(0, inplace = True)
mydf['apoe'].loc[mydf['apoe']>=1] = 1
mydf['apoe'].value_counts()

mydf_famdem = pd.read_csv(dpath + 'Preprocessed_Data/ScoreFeatures/FamilyDementia.csv', usecols=['eid', 'fam_dem'])
mydf_famdem['fam_dem'].value_counts()

mydf = pd.merge(mydf, mydf_educ[['eid', 'educ_years']], how='inner', on=['eid'])
mydf = pd.merge(mydf, mydf_depre[['eid', 'depression']], how='inner', on=['eid'])
mydf = pd.merge(mydf, mydf_famdem[['eid', 'fam_dem']], how='inner', on=['eid'])

mydf_DRS = mydf[['eid', 'dementia_status', 'dementia_years', 'AD_status', 'AD_years',
                 'Age', 'sex', 'educ_years', 'diabetes', 'depression', 'stroke', 'fam_dem', 'apoe']]
mydf_DRS.to_csv(dpath + 'Preprocessed_Data/ScoreData/UKB_DRS.csv')
print('finished')

