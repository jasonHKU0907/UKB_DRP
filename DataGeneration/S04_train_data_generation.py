
import numpy as np
import pandas as pd
import sklearn
import scipy

dpath = '/Volumes/JasonWork/UKB/'

target_df = pd.read_csv(dpath + 'Preprocessed_Data/Dementia_target.csv')

features_df = pd.read_csv(dpath + 'Preprocessed_Data/raw_features.csv', usecols=['Features'])
raw_f = ['eid'] + features_df.iloc[:, 0].tolist()
mydata1 = pd.read_csv(dpath + 'Data/ukb45628.csv', usecols=raw_f)
add_f = ['eid', 'Age', 'APOE4','Sex', 'Townsand', 'BMI', 'Qualification', 'Smoking', 'Alcohal', "pT_0.00000005", "pT_0.000001", "pT_0.000005", "pT_0.00001", "pT_0.00005", "pT_0.0001",
         "pT_0.0005", "pT_0.001", "pT_0.005", "pT_0.01", "pT_0.05", "pT_0.1", "pT_0.5", "pT_1"]
mydata2 = pd.read_csv(dpath + 'Preprocessed_Data/target_full.csv', usecols=add_f)
mydf = pd.merge(mydata1, mydata2, how='inner', on=['eid'])
mydf = pd.merge(target_df, mydf, how='inner', on=['eid'])
rm_f = [f for f in mydf.columns if mydf[f].isnull().sum()/len(mydf)>=0.4]

mydf.drop(rm_f, axis = 1, inplace=True)
mydf.to_csv(dpath + 'Preprocessed_Data/PreSelected_df.csv')

my_f = mydf.columns.tolist()
features_df = pd.DataFrame({'Features': my_f})
dict_f = pd.read_csv(dpath + 'Data/FieldID_selected.csv')
my_lst = pd.merge(features_df, dict_f, how='inner', on=['Features'])
my_f = my_lst['Features']
my_pos_df = mydf.loc[mydf['dementia_status'] == 1]
na_full = [round(mydf[ele].isnull().sum()*100/len(mydf),1) for ele in my_f]
na_pos = [round(my_pos_df[ele].isnull().sum()*100/len(my_pos_df),1) for ele in my_f]
myout = pd.concat((my_lst, pd.DataFrame(na_full), pd.DataFrame(na_pos)), axis=1)
myout.columns = ['Features','Path', 'Field', 'ValueType', 'Units', 'NA_full', 'NA_target']
myout.to_csv(dpath + 'Preprocessed_Data/Screened_Features.csv')
