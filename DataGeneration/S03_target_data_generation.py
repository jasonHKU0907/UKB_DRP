

import numpy as np
import pandas as pd
import sklearn
import scipy

dpath = '/Volumes/JasonWork/UKB/'
target_df = pd.read_csv(dpath + 'Preprocessed_Data/target_full.csv')
target_df = target_df[['eid',
                       'dementia_status',  'dementia_years',
                       'AD_status', 'AD_years',
                       'VD_status', 'VD_years',
                       'stroke_status', 'stroke_years']]

dm_yrspos = target_df.loc[target_df['dementia_years'] >= 0]
dm_yrsneg = target_df.loc[target_df['dementia_years'] < 0]
dm_yrsna = target_df.loc[target_df['dementia_years'].isnull()==True]
dm_yrspos_nostrke = dm_yrspos.drop(dm_yrspos.index[dm_yrspos['stroke_years']<0])
dm_yrspos_nostrke['dementia_status'].value_counts()

dm_yrspos_nostrke.reset_index(inplace =True)
dm_yrspos_nostrke.to_csv(dpath + 'Preprocessed_Data/Dementia_target.csv')

