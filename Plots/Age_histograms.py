

import numpy as np
import pandas as pd
import sklearn
import scipy

dpath = '/Volumes/JasonWork/UKB/'

full_df = pd.read_csv(dpath + 'Preprocessed_Data/target.csv')
sub_df = pd.read_csv(dpath + 'Preprocessed_Data/Dementia_target.csv')

mydf = pd.merge(full_df, sub_df, how='inner', on=['eid'])
mydf['update_date'] = '12/1/2020'
mydf1 = mydf.loc[mydf['Age']<=55]
mydf2 = mydf.loc[(mydf['Age']>55) & (mydf['Age']<=65)]
mydf3 = mydf.loc[mydf['Age']>65]
mydf1.shape
mydf2.shape
mydf3.shape
mydf1['dementia_status_x'].value_counts()
mydf2['dementia_status_x'].value_counts()
mydf3['dementia_status_x'].value_counts()
mydf1['AD_status_x'].value_counts()
mydf2['AD_status_x'].value_counts()
mydf3['AD_status_x'].value_counts()

mydf1.iloc[mydf1['dementia_status_x']==0, 9] = np.nan
fig, ax = plt.subplots(figsize=(10, 5))
ax.plot(mydf1['Age'], alpha = 0.7)
ax.set_xlim(0, 14.5)
ax.set_yticklabels(np.round(np.linspace(0, 0.175, 8), 3), fontsize = 15)
ax.set_xticklabels([0, 2, 4, 6, 8, 10, 12, 14], fontsize = 15)
ax.set_ylabel('Density', fontsize = 24)
ax.set_xlabel('First reported dementia time (years)', fontsize = 24)
fig.tight_layout()


mydf2.iloc[mydf2['dementia_status_x']==0, 9] = np.nan
fig, ax = plt.subplots(figsize=(10, 5))
ax.hist(mydf2['dementia_years_x'], bins=50, density = True, alpha = 0.7)
ax.set_xlim(0, 14.5)
ax.set_yticklabels(np.round(np.linspace(0, 0.175, 8), 3), fontsize = 15)
ax.set_xticklabels([0, 2, 4, 6, 8, 10, 12, 14], fontsize = 15)
ax.set_ylabel('Density', fontsize = 24)
ax.set_xlabel('First reported dementia time (years)', fontsize = 24)
fig.tight_layout()

mydf3.iloc[mydf3['dementia_status_x']==0, 9] = np.nan
fig, ax = plt.subplots(figsize=(10, 5))
ax.hist(mydf3['dementia_years_x'], bins=50, density = True, alpha = 0.7)
ax.set_xlim(0, 14.5)
ax.set_yticklabels(np.round(np.linspace(0, 0.175, 8), 3), fontsize = 15)
ax.set_xticklabels([0, 2, 4, 6, 8, 10, 12, 14], fontsize = 15)
ax.set_ylabel('Density', fontsize = 24)
ax.set_xlabel('First reported dementia time (years)', fontsize = 24)
fig.tight_layout()