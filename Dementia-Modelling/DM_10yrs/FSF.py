
import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedKFold
from Utility.Training_Utilities import *
from lightgbm import LGBMClassifier
import lightgbm as lgb
import matplotlib.pyplot as plt
import seaborn as sns

dpath = '/Volumes/JasonWork/UKB/'
dict_f = pd.read_csv(dpath + 'Data/FieldID_selected1.csv')
fimp_s03 = pd.read_csv(dpath + 'Results/Results_woHES/DM_10yrs/s03_DM_10yrs.csv')
fimp_s03.sort_values(by = 'Cover', ascending = False, inplace = True)
fimp_s04 = pd.read_csv(dpath + 'Results/Results_woHES/DM_10yrs/s04_DM_10yrs.csv')
fimp_s04['f_idx'] = fimp_s04['Unnamed: 0'] + 1
mylabel = match_labels(fimp_s03['Features'], dict_f)
mylabel.iloc[1] = r'ApoE' + ' ' + '$\epsilon 4$'
fimp_tc = pd.DataFrame(zip(fimp_s03['Cover'], mylabel), columns=['Fimp','Feature'])
fimp_tc = pd.concat((fimp_s04[['f_idx', 'AUC_mean', 'AUC0', 'AUC1',
                               'AUC2', 'AUC3', 'AUC4', 'AUC_std']], fimp_tc), axis = 1)
fimp_tc['AUC_lower'] = fimp_tc['AUC_mean'] - 1.96*fimp_tc['AUC_std']
fimp_tc['AUC_upper'] = fimp_tc['AUC_mean'] + 1.96*fimp_tc['AUC_std']

fig, ax = plt.subplots(figsize = (18, 7))
palette = sns.color_palette("Blues",n_colors=len(fimp_tc))
palette.reverse()
sns.barplot(ax=ax, x = "Feature", y = "Fimp", palette=palette, data=fimp_tc.sort_values(by="Fimp", ascending=False))
ax.set_ylim([0, 0.13])
ax.tick_params(axis='y', labelsize=14)
ax.set_xticklabels(fimp_tc['Feature'], rotation=30, fontsize=12, horizontalalignment='right')
nb_f = 10
my_col = ['r']*nb_f + ['k']*(len(fimp_tc)-nb_f)
for ticklabel, tickcolor in zip(plt.gca().get_xticklabels(), my_col):
    ticklabel.set_color(tickcolor)
ax.set_ylabel('Predictor Importance', weight='bold', fontsize=18)
#ax.set_title('10-year incident Dementia', y=1.0, pad=-25, weight='bold', fontsize=24)
ax.set_xlabel('')
ax.grid(which='minor', alpha=0.2, linestyle=':')
ax.grid(which='major', alpha=0.5,  linestyle='--')
ax.set_axisbelow(True)

ax2 = ax.twinx()
#ax2.plot(fimp_tc['f_idx']-1, fimp_tc['AUC0'], 'mediumvioletred', alpha = 0.15, marker='o')
#ax2.plot(fimp_tc['f_idx']-1, fimp_tc['AUC1'], 'mediumvioletred', alpha = 0.15, marker='o')
#ax2.plot(fimp_tc['f_idx']-1, fimp_tc['AUC2'], 'mediumvioletred', alpha = 0.15, marker='o')
#ax2.plot(fimp_tc['f_idx']-1, fimp_tc['AUC3'], 'mediumvioletred', alpha = 0.15, marker='o')
#ax2.plot(fimp_tc['f_idx']-1, fimp_tc['AUC4'], 'mediumvioletred', alpha = 0.15, marker='o')
ax2.plot(np.arange(nb_f+1), fimp_tc['AUC_mean'][:nb_f+1], 'red', alpha = 0.8, marker='o')
ax2.plot(np.arange(nb_f+1, len(fimp_tc)), fimp_tc['AUC_mean'][nb_f+1:], 'black', alpha = 0.8, marker='o')
ax2.plot([nb_f, nb_f+1], fimp_tc['AUC_mean'][nb_f:nb_f+2], 'black', alpha = 0.8, marker='o')
#ax2.plot(fimp_tc['f_idx']-1, fimp_tc['AUC_mean'], 'red', alpha = 0.8, marker='o')
plt.fill_between(fimp_tc['f_idx']-1, fimp_tc['AUC_lower'], fimp_tc['AUC_upper'], color = 'tomato', alpha = 0.2)
ax2.set_ylabel('Cumulative AUC', weight='bold', fontsize=18)
ax2.tick_params(axis='y', labelsize=14)
#ax2.set_yticklabels([0.78, 0.80, 0.82, 0.84, 0.86], fontsize=14)
fig.tight_layout()
plt.xlim([-.6, len(fimp_tc)-.2])
plt.savefig(dpath + 'Results/Results_woHES/DM_10yrs/Cover_Imp_10yrs.png')
