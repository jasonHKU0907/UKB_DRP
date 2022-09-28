


import matplotlib.pyplot as plt
import numpy as np
from scipy import interp
from sklearn.metrics import roc_curve
import pandas as pd


dpath = '/Volumes/JasonWork/UKB/'
y_pred_5yrs = pd.read_csv(dpath + 'Results/Results_AD_Age/Age_55/pred_prob_cv_df.csv')
y_test_5yrs = pd.read_csv(dpath + 'Results/Results_AD_Age/Age_55/test_cv_df.csv')
y_pred_10yrs = pd.read_csv(dpath + 'Results/Results_AD_Age/Age55_65/pred_prob_cv_df.csv')
y_test_10yrs = pd.read_csv(dpath + 'Results/Results_AD_Age/Age55_65/test_cv_df.csv')
y_pred_full = pd.read_csv(dpath + 'Results/Results_AD_Age/Age65_/pred_prob_cv_df.csv')
y_test_full = pd.read_csv(dpath + 'Results/Results_AD_Age/Age65_/test_cv_df.csv')

plt.figure(figsize = (12, 12))
tprs = []
base_fpr = np.linspace(0, 1, 101)

for i in range(1,6):
    y_pred = y_pred_full.iloc[:13350, i]
    y_true = y_test_full.iloc[:13350, i]
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
         label = 'dementia all-time')
plt.fill_between(base_fpr, tprs_lower, tprs_upper, color = 'lightskyblue', alpha = 0.2)



tprs = []
base_fpr = np.linspace(0, 1, 101)

for i in range(1,6):
    y_pred = y_pred_10yrs.iloc[:37710, i]
    y_true = y_test_10yrs.iloc[:37710, i]
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
         label = 'dementia within 10-years')
plt.fill_between(base_fpr, tprs_lower, tprs_upper, color = 'lightsalmon', alpha = 0.2)

tprs = []
base_fpr = np.linspace(0, 1, 101)

for i in range(1,6):
    y_pred = y_pred_5yrs.iloc[:33960, i]
    y_true = y_test_5yrs.iloc[:33960, i]
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
         label = 'dementia within 5-years')
plt.fill_between(base_fpr, tprs_lower, tprs_upper, color = 'mediumspringgreen', alpha = 0.2)


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


plt.savefig(dpath + 'Results/Results_AD_Age/AUC.png')
