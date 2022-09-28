

import pandas as pd
dpath = '/Volumes/JasonWork/UKB/'
my_f_tc = pd.read_csv(dpath + 'Results_Trial3/FI_DM_5yrs_tc.csv')['Features'][:30]
my_f_tg = pd.read_csv(dpath + 'Results_Trial3/FI_DM_5yrs_tg.csv')['Features'][:30]
my_f = list(set(list(my_f_tc) + list(my_f_tg)))
my_f = my_f_tg
df1 = pd.DataFrame({'features': my_f})
df2 = pd.read_csv(dpath + 'Data/FieldID_selected.csv')
my_lst = pd.merge(df1, df2, how='inner', on=['features'])
my_f = my_lst['features']

mydf = pd.read_csv(dpath + 'Preprocessed_Data/Preprocessed_Data.csv')
#mydf = mydf.loc[mydf['dementia_status'] == 1]
my_pos_df = mydf.loc[mydf['dementia_status'] == 1]


mydf = pd.read_csv(dpath + 'Preprocessed_Data/Preprocessed_Data.csv')
mydf1 = mydf.loc[(mydf['dementia_years']>=5) & (mydf['dementia_status']==0)]
mydf2 = mydf.loc[(mydf['dementia_years']<5) & (mydf['dementia_status']==1)]
mydf = pd.concat((mydf1, mydf2), axis = 0)
mydf.reset_index(inplace = True)
my_pos_df = mydf.loc[mydf['dementia_status'] == 1]


na_full = [round(mydf[ele].isnull().sum()*100/len(mydf),1) for ele in my_f]
na_pos = [round(my_pos_df[ele].isnull().sum()*100/len(my_pos_df),1) for ele in my_f]
myout = pd.concat((my_lst, pd.DataFrame(na_full), pd.DataFrame(na_pos)), axis=1)

myout.columns = ['Features', 'Path', 'Field', 'ValueType', 'Units', 'NA_full', 'NA_target']
myout.to_csv(dpath + 'Results_Trial3/Results_DM_5yrs_top30.csv')

















import pandas as pd
dpath = '/Volumes/JasonWork/UKB/'
my_f1 = pd.read_csv(dpath + 'Results/s01_DM_5yrs.csv')['Features'][:50]
my_f2 = pd.read_csv(dpath + 'Results/s01_DM_10yrs.csv')['Features'][:50]
my_f3 = pd.read_csv(dpath + 'Results/s01_DM_full.csv')['Features'][:50]
my_f = list(set(list(my_f1) + list(my_f2) + list(my_f3)))


my_f = pd.read_csv(dpath + 'Results/DM_full/s01_DM_full_top80.csv')['Features']
mydf = pd.read_csv(dpath + 'Preprocessed_Data/Preprocessed_Data.csv')
X = mydf[my_f]
mytbl = X.corr(method='pearson')
mytbl1 = round(mytbl,3)
mytbl1.to_csv(dpath + 'Results/DM_full/s01_DM_full_top80_correlations.csv')
mytbl1.where(((mytbl1<0.5) & (mytbl1>-0.5)), np.nan, inplace = True)
mytbl1.where(((mytbl1>0.5) | (mytbl1<-0.5)), inplace = True)
