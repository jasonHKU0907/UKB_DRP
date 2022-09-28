
import numpy as np
import pandas as pd
import pickle
from sklearn.model_selection import StratifiedKFold
from sklearn.feature_selection import RFECV
import matplotlib.pyplot as plt
from Utility.Evaluation_Utilities import *
from lightgbm import LGBMClassifier
from sklearn.calibration import CalibratedClassifierCV
from sklearn.calibration import calibration_curve
import operator
from scipy.stats import spearmanr
from scipy.cluster import hierarchy
from scipy.spatial.distance import squareform
from collections import defaultdict


dpath = '/Volumes/JasonWork/UKB/'
dict_f = pd.read_csv(dpath + 'Data/FieldID_selected.csv')
my_f = pd.read_csv(dpath + 'Results/DM_10yrs/s01_DM_10yrs.csv')
my_f = my_f['Features'][:50]
mydf = pd.read_csv(dpath + 'Preprocessed_Data/Preprocessed_Data.csv')
mydf['leg_fat_mass'] = (mydf['23112-0.0'] + mydf['23116-0.0'])/2
mydf['leg_fat_percentage'] = (mydf['23115-0.0'] + mydf['23111-0.0'])/2
X = mydf[my_f]
y = mydf['dementia_status']
y[mydf['dementia_years']>10] = 0

def match_labels(f_lst, f_dict):
    f_df = pd.DataFrame({'Features':f_lst})
    merged_df = pd.merge(f_df, f_dict, how='inner', on=['Features'])
    return merged_df['Field']

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

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(20, 10))
dendro = hierarchy.dendrogram(dist_linkage, labels=my_label.tolist(), ax=ax2, leaf_rotation=90)
#dendro = hierarchy.dendrogram(dist_linkage, labels=X.columns.tolist(), ax=ax2, leaf_rotation=90)
dendro_idx = np.arange(0, len(dendro["ivl"]))
ax1.imshow(corr[dendro["leaves"], :][:, dendro["leaves"]])
ax1.set_xticks(dendro_idx)
ax1.set_yticks(dendro_idx)
ax1.set_xticklabels(dendro["ivl"], rotation=90, fontsize=8)
ax1.set_yticklabels(dendro["ivl"], fontsize=8)
fig.tight_layout()
plt.show()


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
myout.to_csv(dpath + 'Results/DM_10yrs/s02-1_DM_10yrs.csv')


















best_params = {'n_estimators': 500,
             'max_depth': 15,
             'num_leaves': 10,
             'subsample': 0.7,
             'learning_rate': 0.01,
             'colsample_bytree': 0.7}

for train_idx, test_idx in mykf.split(X, y):
    X_train, X_test = X.iloc[train_idx,:], X.iloc[test_idx,:]
    y_train, y_test = y.iloc[train_idx], y.iloc[test_idx]
    my_gbm = LGBMClassifier(objective = 'binary', is_unbalance = True, n_jobs = 4,
                            metric = 'auc', verbose = -1, seed = 2022)
    my_gbm.set_params(**best_params)
    calibrate = CalibratedClassifierCV(my_gbm, method='isotonic', cv=5)

calibrate.fit(X_train, y_train)

from sklearn.inspection import permutation_importance

#result = permutation_importance(calibrate, X_train, y_train, n_repeats=10, random_state=2022)
result = permutation_importance(calibrate, X_test, y_test, scoring = 'roc_auc', n_repeats=3, random_state=2022)
sorted_idx = result.importances_mean.argsort()
fig, ax = plt.subplots()
ax.boxplot(result.importances[sorted_idx].T, vert=False, labels=X_test.columns[sorted_idx])
ax.set_title("Permutation Importances (test set)")
fig.tight_layout()
plt.show()


best_params = {'n_estimators': 500,
             'max_depth': 15,
             'num_leaves': 10,
             'subsample': 0.7,
             'learning_rate': 0.01,
             'colsample_bytree': 0.7}

my_gbm = LGBMClassifier(objective='binary', metric='auc',
                        is_unbalance=True, verbosity=-1, seed=2022)
my_gbm.set_params(**best_params)
gb_rfecv = RFECV(estimator=my_gbm, step=3, cv = mykf, scoring='roc_auc', importance_getter='auto')
gb_rfecv.fit(X, y)
gb_rfecv.support_.sum()

plt.figure()
plt.title('Gradient Boost CV score vs No of Features')
plt.xlabel("Number of features selected")
plt.ylabel("Cross validation score (nb of correct classifications)")
plt.plot(range(1, len(gb_rfecv.grid_scores_) + 1), np.mean(gb_rfecv.grid_scores_, axis = 1))
plt.show()

output = pd.DataFrame({'features':gb_rfecv.feature_names_in_, 'support':gb_rfecv.support_})
output['features'][output['support']==True].tolist()

gb_rfecv.ranking_
gb_rfecv.support_.sum()
gb_rfecv.feature_names_in_

dict(zip(gb_rfecv.support_, gb_rfecv.feature_names_in_))





import pandas as pd
dpath = '/Volumes/JasonWork/UKB/'
my_f = pd.read_csv(dpath + 'Results/DM_5yrs/s01_DM_5yrs.csv')
my_f.shape
my_path = my_f['Path'].tolist()
my_path1 = set(my_path)
my_path1 = list(my_path1)
my_path1_df = pd.DataFrame({'Path': my_path1})
my_path1_df.to_csv(dpath + 'Results/Unique_Path.csv')