


import matplotlib.pyplot as plt
import numpy as np
from scipy import interp
from sklearn.metrics import roc_curve
import pandas as pd
#from Utility.Training_Utilities import *
from sklearn.calibration import calibration_curve

def extend(my_array, nb_points):
    if len(my_array) == nb_points:
        pass
    else:
        nb2impute = nb_points - len(my_array)
        impute_array = np.zeros(nb2impute)
        my_array = np.concatenate((impute_array, my_array), axis=0)
    return np.expand_dims(my_array, -1)


def autolabel(rects, ax, x_move, y_move, fontsize, color, weight = 'bold'):
    """Attach a text label above each bar in *rects*, displaying its height."""
    for rect in rects:
        height = rect.get_height()
        ax.annotate('{}'.format(height),
                    xy=(rect.get_x() + rect.get_width() / 2 + x_move, height + y_move),
                    xytext=(0, 3),  # 3 points vertical offset
                    textcoords="offset points",
                    ha='center', va='bottom',
                    fontsize = fontsize,
                    color = color,
                    weight = weight)


dpath = '/Volumes/JasonWork/UKB/'
y_pred_5yrs = pd.read_csv(dpath + 'Results/DM_5yrs/pred_prob_cv_4f_df.csv')
y_test_5yrs = pd.read_csv(dpath + 'Results/DM_5yrs/test_cv_4f_df.csv')
y_pred_10yrs = pd.read_csv(dpath + 'Results/DM_10yrs/pred_prob_cv_4f_df.csv')
y_test_10yrs = pd.read_csv(dpath + 'Results/DM_10yrs/test_cv_4f_df.csv')
y_pred_full = pd.read_csv(dpath + 'Results/DM_full/pred_prob_cv_4f_df.csv')
y_test_full = pd.read_csv(dpath + 'Results/DM_full/test_cv_4f_df.csv')

fig, ax = plt.subplots(nrows=3,ncols=1, figsize = (12, 12))
obs_array, pred_array = np.zeros((10, 1)), np.zeros((10, 1))

for i in range(1,6):
    y_test = y_test_5yrs.iloc[:85037, i]
    y_pred_prob = y_pred_5yrs.iloc[:85037, i]
    obsf, predf = calibration_curve(y_test, y_pred_prob, n_bins=10, strategy = 'quantile')
    obs_array = np.concatenate((obs_array, extend(obsf, nb_points=10)), axis=1)
    pred_array = np.concatenate((pred_array, extend(predf, nb_points=10)), axis=1)

obs_array = 1000*obs_array[:, 1:]
pred_array = 1000*pred_array[:, 1:]

obs_mean = np.round(np.mean(obs_array, axis = 1),2)
pred_mean = np.round(np.mean(pred_array, axis = 1),2)
pred_mean[0] = 0.01
x = np.arange(10) +1
width = 0.35  # the width of the bars
rects1 = ax[0].bar(x - width/2, obs_mean, width, color = 'steelblue', label='observed')
rects2 = ax[0].bar(x + width/2, pred_mean, width, color = 'orange', label='predicted')
ax[0].set_ylabel('Probability ' + r'($\perthousand$)', fontsize=18)
ax[0].set_xticks(x)
ax[2].set_yticks([0, 2, 4, 6, 8, 10, 12])
ax[0].set_ylim(0, 14)
ax[0].legend(fontsize=18)
autolabel(rects1, ax[0], x_move = -0.07, y_move = 0, fontsize = 12, color = 'steelblue')
autolabel(rects2, ax[0], x_move = 0.07, y_move=0, fontsize = 12, color = 'orange')
ax[0].grid(which='major', alpha=0.5,  linestyle='--')


obs_array, pred_array = np.zeros((10, 1)), np.zeros((10, 1))

for i in range(1,6):
    y_test = y_test_10yrs.iloc[:85037, i]
    y_pred_prob = y_pred_10yrs.iloc[:85037, i]
    obsf, predf = calibration_curve(y_test, y_pred_prob, n_bins=10, strategy = 'quantile')
    obs_array = np.concatenate((obs_array, extend(obsf, nb_points=10)), axis=1)
    pred_array = np.concatenate((pred_array, extend(predf, nb_points=10)), axis=1)

obs_array = 1000*obs_array[:, 1:]
pred_array = 1000*pred_array[:, 1:]

obs_mean = np.round(np.mean(obs_array, axis = 1),2)
pred_mean = np.round(np.mean(pred_array, axis = 1),2)
x = np.arange(10) +1
width = 0.35  # the width of the bars
rects1 = ax[1].bar(x - width/2, obs_mean, width, color = 'steelblue', label='observed')
rects2 = ax[1].bar(x + width/2, pred_mean, width, color = 'orange', label='predicted')
ax[1].set_ylabel('Probability ' + r'($\perthousand$)', fontsize=16)
ax[1].set_xticks(x)
ax[1].set_ylim(0, 65)
ax[1].legend(fontsize=18)
autolabel(rects1, ax[1], x_move = -0.07, y_move = 0, fontsize = 12, color = 'steelblue')
autolabel(rects2, ax[1], x_move = 0.07, y_move=0, fontsize = 12, color = 'orange')
ax[1].grid(which='major', alpha=0.5,  linestyle='--')



obs_array, pred_array = np.zeros((10, 1)), np.zeros((10, 1))

for i in range(1,6):
    y_test = y_test_full.iloc[:85037, i]
    y_pred_prob = y_pred_full.iloc[:85037, i]
    obsf, predf = calibration_curve(y_test, y_pred_prob, n_bins=10, strategy = 'quantile')
    obs_array = np.concatenate((obs_array, extend(obsf, nb_points=10)), axis=1)
    pred_array = np.concatenate((pred_array, extend(predf, nb_points=10)), axis=1)

obs_array = 1000*obs_array[:, 1:]
pred_array = 1000*pred_array[:, 1:]

obs_mean = np.round(np.mean(obs_array, axis = 1),2)
pred_mean = np.round(np.mean(pred_array, axis = 1),2)
pred_mean[0] = 0.05
x = np.arange(10) +1
width = 0.35  # the width of the bars
rects1 = ax[2].bar(x - width/2, obs_mean, width, color = 'steelblue', label='observed')
rects2 = ax[2].bar(x + width/2, pred_mean, width, color = 'orange', label='predicted')
ax[2].set_ylabel('Probability ' + r'($\perthousand$)', fontsize=16)
ax[2].set_xticks(x)
ax[2].set_yticks([0, 15, 30, 45, 60, 75, 90], fontsize = 12)
ax[2].set_ylim(0,90)
ax[2].legend(fontsize=18)
autolabel(rects1, ax[2], x_move = -0.07, y_move = -0,  fontsize = 12, color = 'steelblue')
autolabel(rects2, ax[2], x_move = 0.07, y_move = -0, fontsize = 12, color = 'orange')
ax[2].grid(which='major', alpha=0.5,  linestyle='--')

fig.tight_layout()

