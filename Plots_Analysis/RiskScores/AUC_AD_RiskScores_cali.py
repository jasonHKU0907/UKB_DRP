


import matplotlib.pyplot as plt
import numpy as np
from scipy import interp
from sklearn.metrics import roc_curve
import pandas as pd
from Utility.DelongTest import *

dpath = '/Volumes/JasonWork/UKB/'
y_pred_full = pd.read_csv(dpath + 'Results/Results_AD_woHES/AD_full/pred_prob_cv_df.csv')
y_test_full = pd.read_csv(dpath + 'Results/Results_AD_woHES/AD_full/test_cv_df.csv')

y_pred_full = pd.read_csv(dpath + 'Results/Deploy/AD_full/pred_prob_cv_df.csv')
y_test_full = pd.read_csv(dpath + 'Results/Deploy/AD_full/test_cv_df.csv')

y_pred_ukbdrs_apoe = pd.read_csv(dpath + 'Results/Results_RiskScores/AD/UKB_DRS_APOE/pred_prob_cv_cali_df.csv')
y_test_ukbdrs_apoe = pd.read_csv(dpath + 'Results/Results_RiskScores/AD/UKB_DRS_APOE/test_cv_cali_df.csv')

y_pred_ukbdrs_woapoe = pd.read_csv(dpath + 'Results/Results_RiskScores/AD/UKB_DRS/pred_prob_cv_cali_df.csv')
y_test_ukbdrs_woapoe = pd.read_csv(dpath + 'Results/Results_RiskScores/AD/UKB_DRS/test_cv_cali_df.csv')

y_pred_drs = pd.read_csv(dpath + 'Results/Results_RiskScores/AD/DRS/pred_prob_cv_cali_df.csv')
y_test_drs = pd.read_csv(dpath + 'Results/Results_RiskScores/AD/DRS/test_cv_cali_df.csv')

y_pred_caide_apoe = pd.read_csv(dpath + 'Results/Results_RiskScores/AD/CAIDE_APOE/pred_prob_cv_cali_df.csv')
y_test_caide_apoe = pd.read_csv(dpath + 'Results/Results_RiskScores/AD/CAIDE_APOE/test_cv_cali_df.csv')

y_pred_caide_woapoe = pd.read_csv(dpath + 'Results/Results_RiskScores/AD/CAIDE_woAPOE/pred_prob_cv_cali_df.csv')
y_test_caide_woapoe = pd.read_csv(dpath + 'Results/Results_RiskScores/AD/CAIDE_woAPOE/test_cv_cali_df.csv')

aucdf_full = pd.read_csv(dpath + 'Results/Results_AD_woHES/AD_full/s05_AD_full.csv')
aucdf_ukbdrs_apoe = pd.read_csv(dpath + 'Results/Results_RiskScores/AD/UKB_DRS_APOE/results_cali_cv.csv')
aucdf_ukbdrs_woapoe = pd.read_csv(dpath + 'Results/Results_RiskScores/AD/UKB_DRS/results_cali_cv.csv')
aucdf_drs = pd.read_csv(dpath + 'Results/Results_RiskScores/AD/DRS/results_cv.csv')
aucdf_caide_apoe = pd.read_csv(dpath + 'Results/Results_RiskScores/AD/CAIDE_APOE/results_cali_cv.csv')
aucdf_woapoe = pd.read_csv(dpath + 'Results/Results_RiskScores/AD/CAIDE_woAPOE/results_cali_cv.csv')


auc_mean_full, auc_std_full = np.round(aucdf_full['AUC'][0], 3), np.round(aucdf_full['AUC_std'][0], 3)
auc_mean_ukbdrs_apoe, auc_std_ukbdrs_apoe = np.round(aucdf_ukbdrs_apoe['AUC'][0], 3), np.round(aucdf_ukbdrs_apoe['AUC_std'][0], 3)
auc_mean_ukbdrs_woapoe, auc_std_ukbdrs_woapoe = np.round(aucdf_ukbdrs_woapoe['AUC'][0], 3), np.round(aucdf_ukbdrs_woapoe['AUC_std'][0], 3)
auc_mean_drs, auc_std_drs = np.round(aucdf_drs['AUC'][0], 3), np.round(aucdf_drs['AUC_std'][0], 3)
auc_mean_caide_apoe, auc_std_caide_apoe = np.round(aucdf_caide_apoe['AUC'][0], 3), np.round(aucdf_caide_apoe['AUC_std'][0], 3)
auc_mean_caide_woapoe, auc_std_caide_woapoe = np.round(aucdf_woapoe['AUC'][0], 3), np.round(aucdf_woapoe['AUC_std'][0], 3)

Delong_ukb_drs_apoe = np.exp(delong_roc_test(y_test_full.iloc[:,1], y_pred_full.iloc[:,1], y_pred_ukbdrs_apoe.iloc[:,1]))
Delong_ukb_drs_woapoe = np.exp(delong_roc_test(y_test_full.iloc[:,1], y_pred_full.iloc[:,1], y_pred_ukbdrs_woapoe.iloc[:,1]))
Delong_drs = np.exp(delong_roc_test(y_test_full.iloc[:,1], y_pred_full.iloc[:,1], y_pred_drs.iloc[:,1]))
Delong_caide_apoe = np.exp(delong_roc_test(y_test_full.iloc[:,1], y_pred_full.iloc[:,1], y_pred_caide_apoe.iloc[:,1]))
Delong_caide_woapoe = np.exp(delong_roc_test(y_test_full.iloc[:,1], y_pred_full.iloc[:,1], y_pred_caide_woapoe.iloc[:,1]))

legend_full  = r'Our Model                       : ' + str(0.862) + '$\pm$' + str(0.015)
legend_ukbdrs_apoe  = r'UKB-DRS                         : ' + str(auc_mean_ukbdrs_apoe) + '$\pm$' + str(auc_std_ukbdrs_apoe) + ' (p<0.01)'
legend_ukbdrs_woapoe = r'UKB-DRS (w/o ApoE ' + '$\epsilon 4)$' + '  : ' + str(auc_mean_ukbdrs_woapoe) + '$\pm$' + str(auc_std_ukbdrs_woapoe) + ' (p<0.01)'
legend_drs  = r'DRS                                : ' + str(auc_mean_drs) + '$\pm$' + str(auc_std_drs) + ' (p<0.01)'
legend_caide_apoe  = r'CAIDE                             : ' + str(auc_mean_caide_apoe) + '$\pm$' + str(auc_std_caide_apoe) + ' (p<0.01)'
legend_caide_woapoe  = r'CAIDE (w/o ApoE ' + '$\epsilon 4)$' + '      : ' + str(auc_mean_caide_woapoe) + '$\pm$' + str(auc_std_caide_woapoe) + ' (p<0.01)'
legend_anuadri  = r'ANU-ADRI                       : ' + str(0.568) + '$\pm$' + str(0.016) + ' (p<0.01)'



plt.figure(figsize = (12, 12))
tprs = []
base_fpr = np.linspace(0, 1, 101)

for i in range(1,6):
    y_pred = y_pred_full.iloc[:85030, i]
    y_true = y_test_full.iloc[:85030, i]
    fpr, tpr, _ = roc_curve(y_true, y_pred)
    plt.plot(fpr, tpr, 'midnightblue', alpha = 0.15)
    tpr = interp(base_fpr, fpr, tpr)
    tpr[0] = 0.0
    tprs.append(tpr)

tprs = np.array(tprs)
mean_tprs = tprs.mean(axis = 0)
std = tprs.std(axis = 0)
tprs_upper = np.minimum(mean_tprs + 2*std, 1)
tprs_lower = mean_tprs - 2*std
plt.plot(base_fpr, mean_tprs, 'midnightblue', linewidth = 2,
         label = legend_full)
plt.fill_between(base_fpr, tprs_lower, tprs_upper, color = 'lightskyblue', alpha = 0.2)


tprs = []
base_fpr = np.linspace(0, 1, 101)
for i in range(1,6):
    y_pred = y_pred_ukbdrs_apoe.iloc[:85030, i]
    y_true = y_test_ukbdrs_apoe.iloc[:85030, i]
    fpr, tpr, _ = roc_curve(y_true, y_pred)
    plt.plot(fpr, tpr, 'firebrick', alpha = 0.15)
    tpr = interp(base_fpr, fpr, tpr)
    tpr[0] = 0.0
    tprs.append(tpr)

tprs = np.array(tprs)
mean_tprs = tprs.mean(axis = 0)
std = tprs.std(axis = 0)
tprs_upper = np.minimum(mean_tprs + 2*std, 1)
tprs_lower = mean_tprs - 2*std
plt.plot(base_fpr, mean_tprs, 'firebrick', linewidth = 2,
         label = legend_ukbdrs_apoe)
plt.fill_between(base_fpr, tprs_lower, tprs_upper, color = 'salmon', alpha = 0.2)



tprs = []
base_fpr = np.linspace(0, 1, 101)
for i in range(1,6):
    y_pred = y_pred_ukbdrs_woapoe.iloc[:85030, i]
    y_true = y_test_ukbdrs_woapoe.iloc[:85030, i]
    fpr, tpr, _ = roc_curve(y_true, y_pred)
    plt.plot(fpr, tpr, 'orangered', alpha = 0.15)
    tpr = interp(base_fpr, fpr, tpr)
    tpr[0] = 0.0
    tprs.append(tpr)

tprs = np.array(tprs)
mean_tprs = tprs.mean(axis = 0)
std = tprs.std(axis = 0)
tprs_upper = np.minimum(mean_tprs + 2*std, 1)
tprs_lower = mean_tprs - 2*std
plt.plot(base_fpr, mean_tprs, 'orangered', linewidth = 2,
         label = legend_ukbdrs_woapoe)
plt.fill_between(base_fpr, tprs_lower, tprs_upper, color = 'lightsalmon', alpha = 0.2)



tprs = []
base_fpr = np.linspace(0, 1, 101)
for i in range(1,6):
    y_pred = y_pred_drs.iloc[:85030, i]
    y_true = y_test_drs.iloc[:85030, i]
    fpr, tpr, _ = roc_curve(y_true, y_pred)
    plt.plot(fpr, tpr, 'darkgreen', alpha = 0.15)
    tpr = interp(base_fpr, fpr, tpr)
    tpr[0] = 0.0
    tprs.append(tpr)

tprs = np.array(tprs)
mean_tprs = tprs.mean(axis = 0)
std = tprs.std(axis = 0)
tprs_upper = np.minimum(mean_tprs + 2*std, 1)
tprs_lower = mean_tprs - 2*std
plt.plot(base_fpr, mean_tprs, 'darkgreen', linewidth = 2,
         label = legend_drs)
plt.fill_between(base_fpr, tprs_lower, tprs_upper, color = 'lime', alpha = 0.2)


tprs = []
base_fpr = np.linspace(0, 1, 101)

for i in range(1,6):
    y_pred = y_pred_caide_apoe.iloc[:85030, i]
    y_true = y_test_caide_apoe.iloc[:85030, i]
    fpr, tpr, _ = roc_curve(y_true, y_pred)
    plt.plot(fpr, tpr, 'mediumblue', alpha = 0.15)
    tpr = interp(base_fpr, fpr, tpr)
    tpr[0] = 0.0
    tprs.append(tpr)

tprs = np.array(tprs)
mean_tprs = tprs.mean(axis = 0)
std = tprs.std(axis = 0)
tprs_upper = np.minimum(mean_tprs + 2*std, 1)
tprs_lower = mean_tprs - 2*std
plt.plot(base_fpr, mean_tprs, 'mediumblue', linewidth = 2,
         label = legend_caide_apoe)
plt.fill_between(base_fpr, tprs_lower, tprs_upper, color = 'mediumslateblue', alpha = 0.2)


tprs = []
base_fpr = np.linspace(0, 1, 101)

for i in range(1,6):
    y_pred = y_pred_caide_woapoe.iloc[:85030, i]
    y_true = y_test_caide_woapoe.iloc[:85030, i]
    fpr, tpr, _ = roc_curve(y_true, y_pred)
    plt.plot(fpr, tpr, 'dodgerblue', alpha = 0.15)
    tpr = interp(base_fpr, fpr, tpr)
    tpr[0] = 0.0
    tprs.append(tpr)

tprs = np.array(tprs)
mean_tprs = tprs.mean(axis = 0)
std = tprs.std(axis = 0)
tprs_upper = np.minimum(mean_tprs + 2*std, 1)
tprs_lower = mean_tprs - 2*std
plt.plot(base_fpr, mean_tprs, 'dodgerblue', linewidth = 2,
         label = legend_caide_woapoe)
plt.fill_between(base_fpr, tprs_lower, tprs_upper, color = 'cornflowerblue', alpha = 0.2)


from sklearn.model_selection import StratifiedKFold
from sklearn import metrics
mykf = StratifiedKFold(n_splits = 5, random_state = 2022, shuffle = True)

mydf_anuadri = pd.read_csv(dpath + 'Preprocessed_Data/ScoreData/ANU_ADRI.csv')
mydf_anuadri['anuadri_score'].value_counts()
X = mydf_anuadri
y = mydf_anuadri['dementia_status']

tprs, fprs = [], []
for train_idx, test_idx in mykf.split(X, y):
    X_test, y_test = X.iloc[test_idx,:], y.iloc[test_idx]
    negative_y = y_test.value_counts()[0]
    positive_y = y_test.value_counts()[1]
    mytbl = pd.crosstab(X_test['anuadri_score'], y_test)
    mytbl_cum = pd.DataFrame(np.zeros((len(mytbl), 2)))
    for i in range(1, len(mytbl)+ 1):
        mytbl_cum.iloc[i-1, :] = mytbl.iloc[:i,:].sum(axis = 0)
    tpr, fpr = [], []
    for j in np.arange(len(mytbl)):
        fp, tp = mytbl_cum.iloc[j, :]
        tn, fn = negative_y - fp, positive_y - tp
        fpr.append(tp / (tp + fn))
        tpr.append(1 - (tn / (fp + tn)))
    plt.plot(fpr, tpr, 'mediumvioletred', alpha = 0.1)
    tpr[0] = 0.0
    tprs.append(np.array(tpr[:39]))
    fprs.append(np.array(fpr[:39]))
    auc_lst.append(metrics.auc(np.array(fpr[:39]), np.array(tpr[:39])))


mean_tprs = np.array(tprs).mean(axis = 0)
mean_tprs[0], mean_tprs[-1] = 0, 1
mean_fprs = np.array(fprs).mean(axis = 0)
std = np.array(tprs).std(axis = 0)
tprs_upper = np.minimum(mean_tprs + 1.96*std, 1)
tprs_lower = mean_tprs - 1.96*std
plt.plot(mean_fprs, mean_tprs, 'magenta', linewidth = 2,
         label = legend_anuadri)
plt.fill_between(mean_fprs, tprs_lower, tprs_upper, color = 'hotpink', alpha = 0.2)

plt.legend(loc=4, fontsize = 20, labelspacing = 1)

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

plt.savefig(dpath + 'Results/Results_RiskScores/Plots/AUC_AD_RiskScores_cali.png')
