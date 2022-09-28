


import matplotlib.pyplot as plt
import numpy as np
from scipy import interp
from sklearn.metrics import roc_curve
import pandas as pd

dpath = '/Volumes/JasonWork/UKB/'
y_pred_5yrs = pd.read_csv(dpath + 'Results/Results_AD_woHES/AD_5yrs/pred_prob_cv_df.csv')
y_test_5yrs = pd.read_csv(dpath + 'Results/Results_AD_woHES/AD_5yrs/test_cv_df.csv')
y_pred_10yrs = pd.read_csv(dpath + 'Results/Results_AD_woHES/AD_10yrs/pred_prob_cv_df.csv')
y_test_10yrs = pd.read_csv(dpath + 'Results/Results_AD_woHES/AD_10yrs/test_cv_df.csv')
y_pred_full = pd.read_csv(dpath + 'Results/Results_AD_woHES/AD_full/pred_prob_cv_df.csv')
y_test_full = pd.read_csv(dpath + 'Results/Results_AD_woHES/AD_full/test_cv_df.csv')

aucdf_5yrs = pd.read_csv(dpath + 'Results/Results_AD_woHES/AD_5yrs/s05_AD_5yrs.csv')
aucdf_10yrs = pd.read_csv(dpath + 'Results/Results_AD_woHES/AD_10yrs/s05_AD_10yrs.csv')
aucdf_full = pd.read_csv(dpath + 'Results/Results_AD_woHES/AD_full/s05_AD_full.csv')

auc_mean_5yrs, auc_std_5yrs = np.round(aucdf_5yrs['AUC'][0], 3), np.round(aucdf_5yrs['AUC_std'][0], 3)
auc_mean_10yrs, auc_std_10yrs = np.round(aucdf_10yrs['AUC'][0], 3), np.round(aucdf_10yrs['AUC_std'][0], 3)
auc_mean_full, auc_std_full = np.round(aucdf_full['AUC'][0], 3), np.round(aucdf_full['AUC_std'][0], 3)

legend_5yrs  = r'5-year incident AD    : ' + str(auc_mean_5yrs) + '$\pm$' + str(auc_std_5yrs)
legend_10yrs = r'10-year incident AD  : ' + str(auc_mean_10yrs) + '$\pm$' + str(auc_std_10yrs)
legend_full  = r'All incident AD           : ' + str(auc_mean_full)  + '$\pm$' + str(auc_std_full)

plt.figure(figsize = (12, 12))
tprs = []
base_fpr = np.linspace(0, 1, 101)

for i in range(1,6):
    y_pred = y_pred_full.iloc[:85030, i]
    y_true = y_test_full.iloc[:85030, i]
    fpr, tpr, _ = roc_curve(y_true, y_pred)
    plt.plot(fpr, tpr, 'mediumvioletred', alpha = 0.15)
    tpr = interp(base_fpr, fpr, tpr)
    tpr[0] = 0.0
    tprs.append(tpr)

tprs = np.array(tprs)
mean_tprs = tprs.mean(axis = 0)
std = tprs.std(axis = 0)
tprs_upper = np.minimum(mean_tprs + 2*std, 1)
tprs_lower = mean_tprs - 2*std
plt.plot(base_fpr, mean_tprs, 'blue', linewidth = 2,
         label = legend_full)
plt.fill_between(base_fpr, tprs_lower, tprs_upper, color = 'lightskyblue', alpha = 0.2)



tprs = []
base_fpr = np.linspace(0, 1, 101)

for i in range(1,6):
    y_pred = y_pred_10yrs.iloc[:85030, i]
    y_true = y_test_10yrs.iloc[:85030, i]
    fpr, tpr, _ = roc_curve(y_true, y_pred)
    plt.plot(fpr, tpr, 'red', alpha = 0.15)
    tpr = interp(base_fpr, fpr, tpr)
    tpr[0] = 0.0
    tprs.append(tpr)

tprs = np.array(tprs)
mean_tprs = tprs.mean(axis = 0)
std = tprs.std(axis = 0)
tprs_upper = np.minimum(mean_tprs + 2*std, 1)
tprs_lower = mean_tprs - 2*std
plt.plot(base_fpr, mean_tprs, 'red', linewidth = 2,
         label = legend_10yrs)
plt.fill_between(base_fpr, tprs_lower, tprs_upper, color = 'lightsalmon', alpha = 0.2)

tprs = []
base_fpr = np.linspace(0, 1, 101)

for i in range(1,6):
    y_pred = y_pred_5yrs.iloc[:85030, i]
    y_true = y_test_5yrs.iloc[:85030, i]
    fpr, tpr, _ = roc_curve(y_true, y_pred)
    plt.plot(fpr, tpr, 'green', alpha = 0.15)
    tpr = interp(base_fpr, fpr, tpr)
    tpr[0] = 0.0
    tprs.append(tpr)

tprs = np.array(tprs)
mean_tprs = tprs.mean(axis = 0)
std = tprs.std(axis = 0)
tprs_upper = np.minimum(mean_tprs + 2*std, 1)
tprs_lower = mean_tprs - 2*std
plt.plot(base_fpr, mean_tprs, 'green', linewidth = 2,
         label = legend_5yrs)
plt.fill_between(base_fpr, tprs_lower, tprs_upper, color = 'mediumspringgreen', alpha = 0.2)

plt.legend(loc=4, fontsize = 22, labelspacing = 1.5)

plt.plot([0, 1], [0, 1], 'r--')
plt.xlim([-0.0, 1.0])
plt.ylim([-0.0, 1.0])
plt.ylabel('True Positive Rate', fontsize = 20)
plt.xlabel('False Positive Rate', fontsize = 20)

plt.yticks([0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0], fontsize = 16)
plt.xticks([0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0], fontsize = 16)
plt.grid(which='minor', alpha=0.2, linestyle=':')
plt.grid(which='major', alpha=0.5,  linestyle='--')
plt.tight_layout()

plt.savefig(dpath + 'Results/Results_AD_woHES/AD_woHES_AUC.png')
