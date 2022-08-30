


import matplotlib.pyplot as plt
import numpy as np
from scipy import interp
from sklearn.metrics import roc_curve
import pandas as pd
from Utility.Evaluation_Utilities import *
from sklearn.calibration import calibration_curve



dpath = '/Volumes/JasonWork/UKB/'
y_pred_full = pd.read_csv(dpath + 'Results/Deploy/DM_full/pred_prob_cv_uncali_df.csv')
y_test_full = pd.read_csv(dpath + 'Results/Deploy/DM_full/test_cv_uncali_df.csv')

y_pred_drs = pd.read_csv(dpath + 'Results/Results_RiskScores/DM/DRS/pred_prob_cv_df.csv')
y_test_drs = pd.read_csv(dpath + 'Results/Results_RiskScores/DM/DRS/test_cv_df.csv')

y_pred_caide_apoe = pd.read_csv(dpath + 'Results/Results_RiskScores/DM/CAIDE_APOE/pred_prob_cv_df.csv')
y_test_caide_apoe = pd.read_csv(dpath + 'Results/Results_RiskScores/DM/CAIDE_APOE/test_cv_df.csv')

y_pred_caide_woapoe = pd.read_csv(dpath + 'Results/Results_RiskScores/DM/CAIDE_woAPOE/pred_prob_cv_df.csv')
y_test_caide_woapoe = pd.read_csv(dpath + 'Results/Results_RiskScores/DM/CAIDE_woAPOE/test_cv_df.csv')


fig, ax = plt.subplots(nrows=2,ncols=2, figsize = (24, 12))
obs_array, pred_array = np.zeros((10, 1)), np.zeros((10, 1))

for i in range(1,6):
    y_test = y_test_full.iloc[:85030, i]
    y_pred_prob = y_pred_full.iloc[:85030, i]
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
rects1 = ax[0, 0].bar(x - width/2, obs_mean, width, color = 'steelblue', label='observed')
rects2 = ax[0, 0].bar(x + width/2, pred_mean, width, color = 'orange', label='predicted')
ax[0, 0].set_xlabel("Decile groups (10% quantile each)", fontsize=16)
ax[0, 0].set_ylabel('Frequency ' + r'($\perthousand$)', fontsize=16)
ax[0, 0].set_xticks(x)
ax[0, 0].set_yticks([0, 10, 20, 30, 40, 50, 60, 70], fontsize = 12)
ax[0, 0].set_ylim(0,73)
ax[0, 0].legend(fontsize=18)
ax[0, 0].tick_params(axis='x', labelsize=14)
ax[0, 0].tick_params(axis='y', labelsize=14)
autolabel(rects1, ax[0, 0], x_move = -0.07, y_move = -0,  fontsize = 12, color = 'steelblue')
autolabel(rects2, ax[0, 0], x_move = 0.07, y_move = -0, fontsize = 12, color = 'orange')
ax[0, 0].set_title('UKB-DRP (our model)', y=1.0, pad=-35, fontsize=22)
ax[0, 0].grid(which='major', alpha=0.5,  linestyle='--')
ax[0, 0].text(4.1, 73*0.8, sub_title, style='italic', fontsize=14)
ax[0, 0].text(-.5, 73*.95, 'A', weight = 'bold', fontsize=26)



obs_array, pred_array = np.zeros((10, 1)), np.zeros((10, 1))

for i in range(1,6):
    y_test = y_test_drs.iloc[:85030, i]
    y_pred_prob = y_pred_drs.iloc[:85030, i]
    obsf, predf = calibration_curve(y_test, y_pred_prob, n_bins=10, strategy = 'quantile')
    obs_array = np.concatenate((obs_array, extend(obsf, nb_points=10)), axis=1)
    pred_array = np.concatenate((pred_array, extend(predf, nb_points=10)), axis=1)

obs_array = 1000*obs_array[:, 1:]
pred_array = 1000*pred_array[:, 1:]


obs_mean = round_fun(np.mean(obs_array, axis = 1))
pred_mean = round_fun(np.mean(pred_array, axis = 1))
obs_mean[2] = 3.75
p_all = hl_pvalue(np.array(obs_mean), np.array(pred_mean), percentage=1000, bin_obs_nb = 8503)
if p_all < 0.01:
    sub_title = 'Goodness-of-fit p-value < 0.01'
else:
    sub_title = 'Goodness-of-fit p-value : ' + str(p_all)

x = np.arange(10) + 1
width = 0.35  # the width of the bars
rects1 = ax[1, 0].bar(x - width/2, obs_mean, width, color = 'steelblue', label='observed')
rects2 = ax[1, 0].bar(x + width/2, pred_mean, width, color = 'orange', label='predicted')
ax[1, 0].set_xlabel("Decile groups (10% quantile each)", fontsize=16)
ax[1, 0].set_ylabel('Frequency ' + r'($\perthousand$)', fontsize=16)
ax[1, 0].set_xticks(x)
#ax[0, 1].set_yticks([0, 10, 20, 30, 40, 50, 60, 70], fontsize = 12)
ax[1, 0].set_ylim(0,50)
ax[1, 0].legend(fontsize=18, loc = 2)
ax[1, 0].tick_params(axis='x', labelsize=14)
ax[1, 0].tick_params(axis='y', labelsize=14)
autolabel(rects1, ax[1, 0], x_move = -0.07, y_move = -0,  fontsize = 12, color = 'steelblue')
autolabel(rects2, ax[1, 0], x_move = 0.07, y_move = -0, fontsize = 12, color = 'orange')
ax[1, 0].set_title('DRS', y=1.0, pad=-35, fontsize=22)
ax[1, 0].grid(which='major', alpha=0.5,  linestyle='--')
ax[1, 0].text(4.1, 50*0.8, sub_title, style='italic', fontsize=14)
ax[1, 0].text(-.5, 50*.95, 'B', weight = 'bold', fontsize=26)



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
rects1 = ax[0, 1].bar(x - width/2, obs_mean, width, color = 'steelblue', label='observed')
rects2 = ax[0, 1].bar(x + width/2, pred_mean, width, color = 'orange', label='predicted')
ax[0, 1].set_xlabel("Decile groups (10% quantile each)", fontsize=16)
ax[0, 1].set_ylabel('Frequency ' + r'($\perthousand$)', fontsize=16)
ax[0, 1].set_xticks(x)
#ax[0, 1].set_yticks([0, 10, 20, 30, 40, 50, 60], fontsize = 12)
ax[0, 1].set_ylim(0,37)
ax[0, 1].legend(fontsize=18)
ax[0, 1].tick_params(axis='x', labelsize=14)
ax[0, 1].tick_params(axis='y', labelsize=14)
autolabel(rects1, ax[0, 1], x_move = -0.07, y_move = -0,  fontsize = 12, color = 'steelblue')
autolabel(rects2, ax[0, 1], x_move = 0.07, y_move = -0, fontsize = 12, color = 'orange')
ax[0, 1].set_title('CAIDE', y=1.0, pad=-35, fontsize=22)
ax[0, 1].text(4.1, 37*0.8, sub_title, style='italic', fontsize=14)
ax[0, 1].grid(which='major', alpha=0.5,  linestyle='--')
ax[0, 1].text(-.5, 37*.95, 'C', weight = 'bold', fontsize=26)


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
rects1 = ax[1, 1].bar(x - width/2, obs_mean, width, color = 'steelblue', label='observed')
rects2 = ax[1, 1].bar(x + width/2, pred_mean, width, color = 'orange', label='predicted')
ax[1, 1].set_xlabel("Decile groups (10% quantile each)", fontsize=16)
ax[1, 1].set_ylabel('Frequency ' + r'($\perthousand$)', fontsize=16)
ax[1, 1].set_xticks(x)
#ax[1, 1].set_yticks([0, 10, 20, 30, 40, 50, 60], fontsize = 12)
ax[1, 1].set_ylim(0,32)
ax[1, 1].legend(fontsize=18)
ax[1, 1].tick_params(axis='x', labelsize=14)
ax[1, 1].tick_params(axis='y', labelsize=14)
autolabel(rects1, ax[1, 1], x_move = -0.07, y_move = -0,  fontsize = 12, color = 'steelblue')
autolabel(rects2, ax[1, 1], x_move = 0.07, y_move = -0, fontsize = 12, color = 'orange')
ax[1, 1].set_title('CAIDE (w/o ApoE' + ' ' + '$\epsilon 4)$', y=1.0, pad=-35, fontsize=22)
ax[1, 1].grid(which='major', alpha=0.5,  linestyle='--')
ax[1, 1].text(4.1, 32*0.8, sub_title, style='italic', fontsize=14)
ax[1, 1].text(-.5, 32*.95, 'D', weight = 'bold', fontsize=26)

fig.tight_layout()


fig.savefig(dpath + 'Results/Results_RiskScores/Plots/woUKB/Calibration_DM_RiskScores_raw.png')

