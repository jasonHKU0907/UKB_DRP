


import matplotlib.pyplot as plt
import numpy as np
from scipy import interp
from sklearn.metrics import roc_curve
import pandas as pd
from Utility.Evaluation_Utilities import *
from sklearn.calibration import calibration_curve

dpath = '/Volumes/JasonWork/UKB/'
y_pred_dm_5yrs = pd.read_csv(dpath + 'Results/Deploy/DM_5yrs/pred_prob_cv_df.csv')
y_test_dm_5yrs = pd.read_csv(dpath + 'Results/Deploy/DM_5yrs/test_cv_df.csv')
y_pred_dm_10yrs = pd.read_csv(dpath + 'Results/Deploy/DM_10yrs/pred_prob_cv_df.csv')
y_test_dm_10yrs = pd.read_csv(dpath + 'Results/Deploy/DM_10yrs/test_cv_df.csv')
y_pred_dm_all = pd.read_csv(dpath + 'Results/Deploy/DM_full/pred_prob_cv_df.csv')
y_test_dm_all = pd.read_csv(dpath + 'Results/Deploy/DM_full/test_cv_df.csv')

y_pred_ad_5yrs = pd.read_csv(dpath + 'Results/Deploy/AD_5yrs/pred_prob_cv_df.csv')
y_test_ad_5yrs = pd.read_csv(dpath + 'Results/Deploy/AD_5yrs/test_cv_df.csv')
y_pred_ad_10yrs = pd.read_csv(dpath + 'Results/Deploy/AD_10yrs/pred_prob_cv_df.csv')
y_test_ad_10yrs = pd.read_csv(dpath + 'Results/Deploy/AD_10yrs/test_cv_df.csv')
y_pred_ad_all = pd.read_csv(dpath + 'Results/Deploy/AD_full/pred_prob_cv_df.csv')
y_test_ad_all = pd.read_csv(dpath + 'Results/Deploy/AD_full/test_cv_df.csv')

def get_calibrations(y_true_df, y_pred_df):
    obs_array, pred_array = np.zeros((10, 1)), np.zeros((10, 1))
    for i in range(1, 6):
        y_test = y_true_df.iloc[:85031, i]
        y_pred_prob = y_pred_df.iloc[:85031, i]
        obsf, predf = calibration_curve(y_test, y_pred_prob, n_bins=10, strategy='quantile')
        obs_array = np.concatenate((obs_array, extend(obsf, nb_points=10)), axis=1)
        pred_array = np.concatenate((pred_array, extend(predf, nb_points=10)), axis=1)
    obs_array = 1000*obs_array[:, 1:]
    pred_array = 1000*pred_array[:, 1:]
    obs_mean = round_fun(np.mean(obs_array, axis=1))
    pred_mean = round_fun(np.mean(pred_array, axis=1))
    return obs_mean, pred_mean


obs_dm_all, pred_dm_all = get_calibrations(y_test_dm_all, y_pred_dm_all)
obs_dm_10yrs, pred_dm_10yrs = get_calibrations(y_test_dm_10yrs, y_pred_dm_10yrs)
obs_dm_5yrs, pred_dm_5yrs = get_calibrations(y_test_dm_5yrs, y_pred_dm_5yrs)

obs_ad_all, pred_ad_all = get_calibrations(y_test_ad_all, y_pred_ad_all)
obs_ad_10yrs, pred_ad_10yrs = get_calibrations(y_test_ad_10yrs, y_pred_ad_10yrs)
obs_ad_5yrs, pred_ad_5yrs = get_calibrations(y_test_ad_5yrs, y_pred_ad_5yrs)

calibration_df = pd.DataFrame({'obs_dm_all':obs_dm_all, 'pred_dm_all':pred_dm_all,
                               'obs_dm_10yrs': obs_dm_10yrs, 'pred_dm_10yrs': pred_dm_10yrs,
                               'obs_dm_5yrs': obs_dm_5yrs, 'pred_dm_5yrs':pred_dm_5yrs,
                               'obs_ad_all': obs_ad_all, 'pred_ad_all': pred_ad_all,
                               'obs_ad_10yrs': obs_ad_10yrs, 'pred_ad_10yrs':pred_ad_10yrs,
                               'obs_ad_5yrs': obs_ad_5yrs, 'pred_ad_5yrs': pred_ad_5yrs})

calibration_df.to_csv(dpath + 'R_Deploy/Deploy_Models/calibrations.csv')


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
ax[1].set_ylim(0, 52)
ax[1].set_yticks(np.array([0, 10, 20, 30, 40, 50]))
ax[1].legend(fontsize=18)
ax[1].tick_params(axis='x', labelsize=14)
ax[1].tick_params(axis='y', labelsize=14)
autolabel(rects1, ax[1], x_move = -0.07, y_move = 0, fontsize = 12, color = 'steelblue')
autolabel(rects2, ax[1], x_move = 0.07, y_move=0, fontsize = 12, color = 'orange')
ax[1].set_title('10-year incident dementia', y=1.0, pad=-35, fontsize=22)
ax[1].text(4.1, 52*0.75, sub_title, style='italic', fontsize=14)
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
ax[2].set_ylim(0, 12.5)
ax[2].legend(fontsize=18)
ax[2].tick_params(axis='x', labelsize=14)
ax[2].tick_params(axis='y', labelsize=14)
autolabel(rects1, ax[2], x_move = -0.07, y_move = 0, fontsize = 12, color = 'steelblue')
autolabel(rects2, ax[2], x_move = 0.07, y_move=0, fontsize = 12, color = 'orange')
ax[2].set_title('5-year incident dementia', y=1.0, pad=-35, fontsize=22)
ax[2].text(4.1, 12.5*0.75, sub_title, style='italic', fontsize=14)
ax[2].grid(which='major', alpha=0.5,  linestyle='--')


fig.tight_layout()

fig.savefig(dpath + 'Results/Deploy/Calibration_DM2DM.png')

