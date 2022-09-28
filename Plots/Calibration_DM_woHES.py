


import matplotlib.pyplot as plt
import numpy as np
from scipy import interp
from sklearn.metrics import roc_curve
import pandas as pd
from Utility.Evaluation_Utilities import *
from sklearn.calibration import calibration_curve

dpath = '/Volumes/JasonWork/UKB/'
y_pred_5yrs = pd.read_csv(dpath + 'Results/Results_woHES/DM_5yrs/pred_prob_cv_df.csv')
y_test_5yrs = pd.read_csv(dpath + 'Results/Results_woHES/DM_5yrs/test_cv_df.csv')
y_pred_10yrs = pd.read_csv(dpath + 'Results/Results_woHES/DM_10yrs/pred_prob_cv_df.csv')
y_test_10yrs = pd.read_csv(dpath + 'Results/Results_woHES/DM_10yrs/test_cv_df.csv')
y_pred_full = pd.read_csv(dpath + 'Results/Results_woHES/DM_full/pred_prob_cv_df.csv')
y_test_full = pd.read_csv(dpath + 'Results/Results_woHES/DM_full/test_cv_df.csv')

fig, ax = plt.subplots(nrows=3,ncols=1, figsize = (12, 12))
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
sub_title = 'Goodness-of-fit p-value : ' + str(p_all)

x = np.arange(10) +1
width = 0.35  # the width of the bars
rects1 = ax[0].bar(x - width/2, obs_mean, width, color = 'steelblue', label='observed')
rects2 = ax[0].bar(x + width/2, pred_mean, width, color = 'orange', label='predicted')
ax[0].set_xlabel("Decile groups (10% quantile each)", fontsize=16)
ax[0].set_ylabel('Frequency ' + r'($\perthousand$)', fontsize=16)
ax[0].set_xticks(x)
ax[0].set_yticks([0, 10, 20, 30, 40, 50, 60, 70], fontsize = 12)
ax[0].set_ylim(0,75)
ax[0].legend(fontsize=18)
ax[0].tick_params(axis='x', labelsize=14)
ax[0].tick_params(axis='y', labelsize=14)
autolabel(rects1, ax[0], x_move = -0.07, y_move = -0,  fontsize = 12, color = 'steelblue')
autolabel(rects2, ax[0], x_move = 0.07, y_move = -0, fontsize = 12, color = 'orange')
ax[0].set_title('All incident dementia', y=1.0, pad=-35, fontsize=22)
ax[0].grid(which='major', alpha=0.5,  linestyle='--')
ax[0].text(4.1, 75*0.75, sub_title, style='italic', fontsize=14)



obs_array, pred_array = np.zeros((10, 1)), np.zeros((10, 1))

for i in range(1,6):
    y_test = y_test_10yrs.iloc[:85030, i]
    y_pred_prob = y_pred_10yrs.iloc[:85030, i]
    obsf, predf = calibration_curve(y_test, y_pred_prob, n_bins=10, strategy = 'quantile')
    obs_array = np.concatenate((obs_array, extend(obsf, nb_points=10)), axis=1)
    pred_array = np.concatenate((pred_array, extend(predf, nb_points=10)), axis=1)

obs_array = 1000*obs_array[:, 1:]
pred_array = 1000*pred_array[:, 1:]

obs_mean = round_fun(np.mean(obs_array, axis = 1))
pred_mean = round_fun(np.mean(pred_array, axis = 1))
p_all = hl_pvalue(np.array(obs_mean), np.array(pred_mean), percentage=1000, bin_obs_nb = 8503)
sub_title = 'Goodness-of-fit p-value : ' + str(p_all)

x = np.arange(10) +1
width = 0.35  # the width of the bars
rects1 = ax[1].bar(x - width/2, obs_mean, width, color = 'steelblue', label='observed')
rects2 = ax[1].bar(x + width/2, pred_mean, width, color = 'orange', label='predicted')
ax[1].set_ylabel('Frequency ' + r'($\perthousand$)', fontsize=16)
ax[1].set_xlabel("Decile groups (10% quantile each)", fontsize=16)
ax[1].set_xticks(x, fontsize = 14)
ax[1].set_ylim(0, 55)
ax[1].set_yticks(np.array([0, 10, 20, 30, 40, 50]))
ax[1].legend(fontsize=18)
ax[1].tick_params(axis='x', labelsize=14)
ax[1].tick_params(axis='y', labelsize=14)
autolabel(rects1, ax[1], x_move = -0.07, y_move = 0, fontsize = 12, color = 'steelblue')
autolabel(rects2, ax[1], x_move = 0.07, y_move=0, fontsize = 12, color = 'orange')
ax[1].set_title('10-year incident dementia', y=1.0, pad=-35, fontsize=22)
ax[1].text(4.1, 55*0.75, sub_title, style='italic', fontsize=14)
ax[1].grid(which='major', alpha=0.5,  linestyle='--')





obs_array, pred_array = np.zeros((10, 1)), np.zeros((10, 1))

for i in range(1,6):
    y_test = y_test_5yrs.iloc[:85030, i]
    y_pred_prob = y_pred_5yrs.iloc[:85030, i]
    obsf, predf = calibration_curve(y_test, y_pred_prob, n_bins=10, strategy = 'quantile')
    obs_array = np.concatenate((obs_array, extend(obsf, nb_points=10)), axis=1)
    pred_array = np.concatenate((pred_array, extend(predf, nb_points=10)), axis=1)

obs_array = 1000*obs_array[:, 1:]
pred_array = 1000*pred_array[:, 1:]

obs_mean = round_fun(np.mean(obs_array, axis = 1))
pred_mean = round_fun(np.mean(pred_array, axis = 1))
p_all = hl_pvalue(np.array(obs_mean), np.array(pred_mean), percentage=1000, bin_obs_nb = 8503)
sub_title = 'Goodness-of-fit p-value : ' + str(p_all)

#pred_mean[0] = 0.01
x = np.arange(10) +1
width = 0.35  # the width of the bars
rects1 = ax[2].bar(x - width/2, obs_mean, width, color = 'steelblue', label='observed')
rects2 = ax[2].bar(x + width/2, pred_mean, width, color = 'orange', label='predicted')
ax[2].set_xlabel("Decile groups (10% quantile each)", fontsize=16)
ax[2].set_ylabel('Frequency ' + r'($\perthousand$)', fontsize=18)
ax[2].set_xticks(x)
ax[2].set_yticks(np.array([0, 2, 4, 6, 8, 10, 12]))
ax[2].set_ylim(0, 13)
ax[2].legend(fontsize=18)
ax[2].tick_params(axis='x', labelsize=14)
ax[2].tick_params(axis='y', labelsize=14)
autolabel(rects1, ax[2], x_move = -0.07, y_move = 0, fontsize = 12, color = 'steelblue')
autolabel(rects2, ax[2], x_move = 0.07, y_move=0, fontsize = 12, color = 'orange')
ax[2].set_title('5-year incident dementia', y=1.0, pad=-35, fontsize=22)
ax[2].text(4.1, 13*0.75, sub_title, style='italic', fontsize=14)
ax[2].grid(which='major', alpha=0.5,  linestyle='--')

fig.tight_layout()

fig.savefig(dpath + 'Results/Results_woHES/DM_woHES_Calibrate.png')

