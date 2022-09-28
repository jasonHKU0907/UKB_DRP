

import pandas as pd

dpath = '/Volumes/JasonWork/UKB/'
mydf = pd.read_csv(dpath + 'Preprocessed_Data/target.csv')
rm_eid = pd.read_csv(dpath + 'Data/w19542_20220222.csv')['eid'].tolist()
rm_idx = [idx for idx in range(len(mydf)) if mydf['eid'][idx] in rm_eid]
mydf.drop(rm_idx, axis = 0, inplace = True)

prs_df = pd.read_csv(dpath + 'Data/PRS_base_IGAP_stage_1_all_20211126.txt', sep=" ")
prs_df = prs_df.rename(columns = {'IID': 'eid'}, inplace = False)
target_df = pd.merge(mydf, prs_df, how='left', on=['eid'])

target_df.to_csv(dpath + 'Preprocessed_Data/target_full.csv')
