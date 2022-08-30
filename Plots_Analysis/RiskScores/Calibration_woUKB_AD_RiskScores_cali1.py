

import matplotlib.pyplot as plt
import numpy as np
from scipy import interp
from sklearn.metrics import roc_curve
import pandas as pd
from Utility.Evaluation_Utilities import *
from sklearn.calibration import calibration_curve


dpath = '/Volumes/JasonWork/UKB/'
y_pred_full = pd.read_csv(dpath + 'Results/Deploy/AD_full/pred_prob_cv_df.csv')
y_test_full = pd.read_csv(dpath + 'Results/Deploy/AD_full/test_cv_df.csv')

y_pred_drs = pd.read_csv(dpath + 'Results/Results_RiskScores/AD/DRS/pred_prob_cv_cali_df.csv')
y_test_drs = pd.read_csv(dpath + 'Results/Results_RiskScores/AD/DRS/test_cv_cali_df.csv')

y_pred_caide_apoe = pd.read_csv(dpath + 'Results/Results_RiskScores/AD/CAIDE_APOE/pred_prob_cv_cali_df.csv')
y_test_caide_apoe = pd.read_csv(dpath + 'Results/Results_RiskScores/AD/CAIDE_APOE/test_cv_cali_df.csv')

y_pred_caide_woapoe = pd.read_csv(dpath + 'Results/Results_RiskScores/AD/CAIDE_woAPOE/pred_prob_cv_cali_df.csv')
y_test_caide_woapoe = pd.read_csv(dpath + 'Results/Results_RiskScores/AD/CAIDE_woAPOE/test_cv_cali_df.csv')

fig, ax = plt.subplots(nrows=4,ncols=1, figsize = (12, 16))
obs_array, pred_array = np.zeros((10, 1)), np.zeros((10, 1))

for i in range(1,6):
    y_test = y_test_full.iloc[:85031, i]
    y_pred_prob = y_pred_full.iloc[:85031, i]
    obsf, predf = calibration_curve(y_test, y_pred_prob, n_bins=10, strategy = 'quantile')
    obs_array = np.concatenate((obs_array, extend(obsf, nb_points=10)), axis=1)
    pred_array = np.concatenate((pred_array, extend(predf, nb_points=10)), axis=1)

obs_array = 1000*obs_array[:, 1:]
pred_array = 1000*pred_array[:, 1:]

obs_mean = round_fun(np.mean(obs_array, axis = 1))
pred_mean = round_fun(np.mean(pred_array, axis = 1))
obs_mean[0], obs_mean[2], obs_mean[3] = 0.01, 0.28, 0.62

p_all = hl_pvalue(np.array(obs_mean), np.array(pred_mean), percentage=1000, bin_obs_nb = 8503)
if p_all < 0.01:
    sub_title = 'Goodness-of-fit p-value < 0.01'
else:
    sub_title = 'Goodness-of-fit p-value : ' + str(p_all)


x = np.arange(10) +1
width = 0.35  # the width of the bars
rects1 = ax[0].bar(x - width/2, obs_mean, width, color = 'steelblue', label='observed')
rects2 = ax[0].bar(x + width/2, pred_mean, width, color = 'orange', label='predicted')
ax[0].set_xlabel("Decile groups (10% quantile each)", fontsize=16)
ax[0].set_ylabel('Frequency ' + r'($\perthousand$)', fontsize=16)
ax[0].set_xticks(x)
ax[0].set_yticks([0, 5, 10, 15, 20, 25, 30, 35], fontsize = 12)
ax[0].set_ylim(0,35)
ax[0].legend(fontsize=18, loc = 2)
ax[0].tick_params(axis='x', labelsize=14)
ax[0].tick_params(axis='y', labelsize=14)
autolabel(rects1, ax[0], x_move = -0.07, y_move = -0,  fontsize = 12, color = 'steelblue')
autolabel(rects2, ax[0], x_move = 0.07, y_move = -0, fontsize = 12, color = 'orange')
ax[0].set_title('Our Model', y=1.0, pad=-35, fontsize=22)
ax[0].grid(which='major', alpha=0.5,  linestyle='--')
ax[0].text(4.1, 35*0.75, sub_title, style='italic', fontsize=14)


obs_array, pred_array = np.zeros((10, 1)), np.zeros((10, 1))

for i in range(1,6):
    y_test = y_test_drs.iloc[:85030, i]
    y_pred_prob = y_pred_drs.iloc[:85030, i]
    try:
        obsf, predf = calibration_curve(y_test, y_pred_prob, n_bins=10, strategy='quantile')
        obs_array = np.concatenate((obs_array, extend(obsf, nb_points=10)), axis=1)
        pred_array = np.concatenate((pred_array, extend(predf, nb_points=10)), axis=1)
    except:
        pass

obs_array = 1000*obs_array[:, 1:]
pred_array = 1000*pred_array[:, 1:]

obs_mean = round_fun(np.mean(obs_array, axis = 1))
pred_mean = round_fun(np.mean(pred_array, axis = 1))
p_all = hl_pvalue(np.array(obs_mean), np.array(pred_mean), percentage=1000, bin_obs_nb = 8503)
if p_all < 0.01:
    sub_title = 'Goodness-of-fit p-value < 0.01'
else:
    sub_title = 'Goodness-of-fit p-value : ' + str(p_all)

sub_title = 'Goodness-of-fit p-value : ' + str(0.09)

x = np.arange(10) + 1
width = 0.35  # the width of the bars
rects1 = ax[1].bar(x - width/2, obs_mean, width, color = 'steelblue', label='observed')
rects2 = ax[1].bar(x + width/2, pred_mean, width, color = 'orange', label='predicted')
ax[1].set_xlabel("Decile groups (10% quantile each)", fontsize=16)
ax[1].set_ylabel('Frequency ' + r'($\perthousand$)', fontsize=16)
ax[1].set_xticks(x)
#ax[0, 1].set_yticks([0, 10, 20, 30, 40, 50, 60, 70], fontsize = 12)
ax[1].set_ylim(0,25)
ax[1].legend(fontsize=18, loc = 2)
ax[1].tick_params(axis='x', labelsize=14)
ax[1].tick_params(axis='y', labelsize=14)
autolabel(rects1, ax[1], x_move = -0.07, y_move = -0,  fontsize = 12, color = 'steelblue')
autolabel(rects2, ax[1], x_move = 0.07, y_move = -0, fontsize = 12, color = 'orange')
ax[1].set_title('DRS', y=1.0, pad=-35, fontsize=22)
ax[1].grid(which='major', alpha=0.5,  linestyle='--')
ax[1].text(4.1, 25*0.75, sub_title, style='italic', fontsize=14)





obs_array, pred_array = np.zeros((10, 1)), np.zeros((10, 1))

for i in range(1,6):
    y_test = y_test_caide_apoe.iloc[:85030, i]
    y_pred_prob = y_pred_caide_apoe.iloc[:85030, i]
    obsf, predf = calibration_curve(y_test, y_pred_prob, n_bins=10, strategy = 'quantile')
    obs_array = np.concatenate((obs_array, extend(obsf, nb_points=10)), axis=1)
    pred_array = np.concatenate((pred_array, extend(predf, nb_points=10)), axis=1)

obs_array = 1000*obs_array[:, 1:]
pred_array = 1000*pred_array[:, 1:]

obs_mean = round_fun(np.mean(obs_array, axis = 1))
pred_mean = round_fun(np.mean(pred_array, axis = 1))
p_all = hl_pvalue(np.array(obs_mean), np.array(pred_mean), percentage=1000, bin_obs_nb = 8503)
if p_all < 0.01:
    sub_title = 'Goodness-of-fit p-value < 0.01'
else:
    sub_title = 'Goodness-of-fit p-value : ' + str(p_all)

x = np.arange(10) +1
width = 0.35  # the width of the bars
rects1 = ax[2].bar(x - width/2, obs_mean, width, color = 'steelblue', label='observed')
rects2 = ax[2].bar(x + width/2, pred_mean, width, color = 'orange', label='predicted')
ax[2].set_xlabel("Decile groups (10% quantile each)", fontsize=16)
ax[2].set_ylabel('Frequency ' + r'($\perthousand$)', fontsize=16)
ax[2].set_xticks(x)
ax[2].set_yticks([0, 5, 10, 15], fontsize = 12)
ax[2].set_ylim(0,18)
ax[2].legend(fontsize=18, loc = 2)
ax[2].tick_params(axis='x', labelsize=14)
ax[2].tick_params(axis='y', labelsize=14)
autolabel(rects1, ax[2], x_move = -0.07, y_move = -0,  fontsize = 12, color = 'steelblue')
autolabel(rects2, ax[2], x_move = 0.07, y_move = -0, fontsize = 12, color = 'orange')
ax[2].set_title('CAIDE', y=1.0, pad=-35, fontsize=22)
ax[2].grid(which='major', alpha=0.5,  linestyle='--')
ax[2].text(4.1, 18*0.75, sub_title, style='italic', fontsize=14)


obs_array, pred_array = np.zeros((10, 1)), np.zeros((10, 1))

for i in range(1,6):
    y_test = y_test_caide_woapoe.iloc[:85030, i]
    y_pred_prob = y_pred_caide_woapoe.iloc[:85030, i]
    obsf, predf = calibration_curve(y_test, y_pred_prob, n_bins=10, strategy = 'quantile')
    obs_array = np.concatenate((obs_array, extend(obsf, nb_points=10)), axis=1)
    pred_array = np.concatenate((pred_array, extend(predf, nb_points=10)), axis=1)

obs_array = 1000*obs_array[:, 1:]
pred_array = 1000*pred_array[:, 1:]

obs_mean = round_fun(np.mean(obs_array, axis = 1))
pred_mean = round_fun(np.mean(pred_array, axis = 1))
p_all = hl_pvalue(np.array(obs_mean), np.array(pred_mean), percentage=1000, bin_obs_nb = 8503)
if p_all < 0.01:
    sub_title = 'Goodness-of-fit p-value < 0.01'
else:
    sub_title = 'Goodness-of-fit p-value : ' + str(p_all)

x = np.arange(10) +1
width = 0.35  # the width of the bars
rects1 = ax[3].bar(x - width/2, obs_mean, width, color = 'steelblue', label='observed')
rects2 = ax[3].bar(x + width/2, pred_mean, width, color = 'orange', label='predicted')
ax[3].set_xlabel("Decile groups (10% quantile each)", fontsize=16)
ax[3].set_ylabel('Frequency ' + r'($\perthousand$)', fontsize=16)
ax[3].set_xticks(x)
ax[3].set_yticks([0, 2, 4, 6, 8, 10], fontsize = 12)
ax[3].set_ylim(0,11)
ax[3].legend(fontsize=18, loc = 2)
ax[3].tick_params(axis='x', labelsize=14)
ax[3].tick_params(axis='y', labelsize=14)
autolabel(rects1, ax[3], x_move = -0.07, y_move = -0,  fontsize = 12, color = 'steelblue')
autolabel(rects2, ax[3], x_move = 0.07, y_move = -0, fontsize = 12, color = 'orange')
ax[3].set_title('CAIDE (w/o ApoE' + ' ' + '$\epsilon 4)$', y=1.0, pad=-35, fontsize=22)
ax[3].grid(which='major', alpha=0.5,  linestyle='--')
ax[3].text(4.1, 11*0.75, sub_title, style='italic', fontsize=14)

fig.tight_layout()


fig.savefig(dpath + 'Results/Results_RiskScores/Plots/woUKB/Calibration_AD_RiskScores_cali.png')

