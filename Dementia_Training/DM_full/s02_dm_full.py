

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from Utility.Training_Utilities import *
from scipy.cluster import hierarchy
from scipy.spatial.distance import squareform
from collections import defaultdict

dpath = '/Volumes/JasonWork/UKB/'
dict_f = pd.read_csv(dpath + 'Data/FieldID_selected.csv')
my_f = pd.read_csv(dpath + 'Results/Results_woHES/DM_full/s01_DM_full.csv')['Features'][:50].tolist()
mydf = pd.read_csv(dpath + 'Preprocessed_Data/Preprocessed_Data.csv')
X = mydf[my_f]
y = mydf['dementia_status']
my_label = match_labels(my_f, dict_f)

corr = np.array(X.corr(method='spearman'))
#corr = np.array(X.corr(method='pearson'))
corr = np.nan_to_num(corr)
corr = (corr + corr.T) / 2
np.fill_diagonal(corr, 1)
# We convert the correlation matrix to a distance matrix before performing
# hierarchical clustering using Ward's linkage.
distance_matrix = 1 - np.abs(corr)
dist_linkage = hierarchy.ward(squareform(distance_matrix))

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(18, 10))
dendro = hierarchy.dendrogram(dist_linkage, labels=my_label.tolist(), ax=ax2)
ax2.set_xticklabels(dendro["ivl"], rotation=60, fontsize=8, horizontalalignment='right')
dendro_idx = np.arange(0, len(dendro["ivl"]))
ax1.imshow(corr[dendro["leaves"], :][:, dendro["leaves"]])
ax1.set_xticks(dendro_idx)
ax1.set_yticks(dendro_idx)
ax1.set_xticklabels(dendro["ivl"], rotation=60, fontsize=8, horizontalalignment='right')
ax1.set_yticklabels(dendro["ivl"], fontsize=8)
fig.tight_layout()
plt.show()
plt.savefig(dpath + 'Results/Results_woHES/DM_full/s02_DM_full.png')


cluster_ids = hierarchy.fcluster(dist_linkage, .75, criterion="distance")
cluster_id_to_feature_ids = defaultdict(list)
for idx, cluster_id in enumerate(cluster_ids):
    cluster_id_to_feature_ids[cluster_id].append(idx)
selected_idx = [v[0] for v in cluster_id_to_feature_ids.values()]
selected_f = X.columns[selected_idx]
selected_f = pd.DataFrame({'Features':selected_f})
dict_f = pd.read_csv(dpath + 'Data/FieldID_selected.csv')
my_lst = pd.merge(selected_f, dict_f, how='inner', on=['Features'])
my_f = my_lst['Features']
my_pos_df = mydf.loc[mydf['dementia_status'] == 1]
na_full = [round(mydf[ele].isnull().sum()*100/len(mydf),1) for ele in my_f]
na_pos = [round(my_pos_df[ele].isnull().sum()*100/len(my_pos_df),1) for ele in my_f]
myout = pd.concat((my_lst, pd.DataFrame(na_full), pd.DataFrame(na_pos)), axis=1)
myout.columns = ['Features', 'Path', 'Field', 'ValueType', 'Units', 'NA_full', 'NA_target']
myout.to_csv(dpath + 'Results/Results_woHES/DM_full/s02_DM_full.csv')

print('Finished')


