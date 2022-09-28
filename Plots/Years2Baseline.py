

import numpy as np
import pandas as pd
import sklearn
import scipy

dpath = '/Volumes/JasonWork/UKB/'

full_df = pd.read_csv(dpath + 'Preprocessed_Data/target.csv')
sub_df = pd.read_csv(dpath + 'Preprocessed_Data/Dementia_target.csv')

mydf = pd.merge(full_df, sub_df, how='inner', on=['eid'])
mydf['update_date'] = '12/1/2020'

def get_days_intervel(start_date_var, end_date_var, df, type):
    start_date = pd.to_datetime(df[start_date_var])
    end_date = pd.to_datetime(df[end_date_var])
    nb_of_dates = start_date.shape[0]
    days = [(end_date[i] - start_date[i]).days for i in range(nb_of_dates)]
    months = [ele/30 for ele in days]
    years = [ele/365 for ele in days]
    colnames = [type+'_days', type+'_months', type+'_years']
    return pd.DataFrame({colnames[0]:days, colnames[1]:months, colnames[2]:years})

tmp = get_days_intervel('Re_date.x', 'update_date', mydf, type = 'fl')
tmp['fl_years'].describe()

import matplotlib.pyplot as plt

fig, ax = plt.subplots(figsize=(10, 5))
ax.hist(tmp['fl_years'], bins=100, density = True, alpha = 0.7)
ax.set_xlim(10, 15)
ax.set_yticklabels(np.round(np.linspace(0, 0.5, 6), 3), fontsize = 15)
ax.set_xticklabels([10, 11, 12, 13, 14, 15], fontsize = 15)
ax.set_ylabel('Density', fontsize = 24)
ax.set_xlabel('Observation time (years)', fontsize = 24)
fig.tight_layout()


mydf.iloc[mydf['dementia_status_x']==0, 9] = np.nan
fig, ax = plt.subplots(figsize=(10, 5))
ax.hist(mydf['dementia_years_x'], bins=100, density = True, alpha = 0.7)
ax.set_xlim(0, 14.5)
ax.set_yticklabels(np.round(np.linspace(0, 0.175, 8), 3), fontsize = 15)
ax.set_xticklabels([0, 2, 4, 6, 8, 10, 12, 14], fontsize = 15)
ax.set_ylabel('Density', fontsize = 24)
ax.set_xlabel('First reported dementia time (years)', fontsize = 24)
fig.tight_layout()

mydf.iloc[mydf['AD_status_x']==0, 16] = np.nan
mydf['AD_years_x'][mydf['AD_years_x']<0] = mydf['dementia_years_x'][mydf['AD_years_x']<0]
fig, ax = plt.subplots(figsize=(10, 5))
ax.hist(mydf['AD_years_x'], bins=100, density = True, alpha = 0.7)
ax.set_xlim(0, 14.5)
ax.set_yticklabels(np.round(np.linspace(0, 0.2, 9), 3), fontsize = 15)
ax.set_xticklabels([0, 2, 4, 6, 8, 10, 12, 14], fontsize = 15)
ax.set_ylabel('Density', fontsize = 24)
ax.set_xlabel("First reported Alzheimer's disease time (years)", fontsize = 24)
fig.tight_layout()

#ax.vlines(5, 0, .175, linestyle="dashed")
#ax.vlines(10, 0, .175, linestyle="dashed")
#ax.hlines(y, 0, x, linestyle="dashed")

y = mydf['AD_status_x'].copy()
y[mydf['AD_years_x']>10] = 0
y.sum()