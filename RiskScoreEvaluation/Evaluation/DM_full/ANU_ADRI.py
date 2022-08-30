


import matplotlib.pyplot as plt
import numpy as np
from scipy import interp
from sklearn.metrics import roc_curve
import pandas as pd


dpath = '/Volumes/JasonWork/UKB/'

mydf_anuadri = pd.read_csv(dpath + 'Preprocessed_Data/ScoreData/ANU_ADRI.csv')
mydf_anuadri['anuadri_score'].value_counts()
negative_y = mydf_anuadri['dementia_status'].value_counts()[0]
positive_y = mydf_anuadri['dementia_status'].value_counts()[1]

mytbl = pd.crosstab(mydf_anuadri['anuadri_score'], mydf_anuadri['dementia_status'])
mytbl_cum = pd.DataFrame(np.zeros((len(mytbl), 2)))

for i in range(1, len(mytbl)+ 1):
    mytbl_cum.iloc[i-1, :] = mytbl.iloc[:i,:].sum(axis = 0)

tpr_lst, tnr_lst = [], []

for i in np.arange(len(mytbl)):
    fp, tp = mytbl_cum.iloc[i, :]
    tn, fn = negative_y - fp, positive_y - tp
    tpr_lst.append(tp / (tp + fn))
    tnr_lst.append(tn / (fp + tn))

fpr_lst = [1-ele for ele in tnr_lst]

plt.plot(tpr_lst, fpr_lst)
plt.xlim([0, 1])
plt.ylim([0, 1])
plt.plot([0, 1], [0, 1], 'r--')

from sklearn import metrics
metrics.auc(tpr_lst, fpr_lst)
