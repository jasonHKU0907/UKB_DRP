

import numpy as np
import pandas as pd
import sklearn
import scipy

dpath = '/Volumes/JasonWork/UKB/'

target_df = pd.read_csv(dpath + 'Preprocessed_Data/Dementia_target.csv')
mydata = pd.read_csv(dpath + 'Preprocessed_Data/PreSelected_df.csv')
my_f_full = pd.read_csv(dpath + 'Preprocessed_Data/Screened_Features.csv')['Features'].tolist()
my_f_full = my_f_full + ['eid', 'dementia_status', 'dementia_years', 'AD_status',
                         'AD_years', 'VD_status', 'VD_years', 'stroke_status']
my_f_rm = pd.read_csv(dpath + 'Preprocessed_Data/Manual2Remove_Features.csv')['Features'].tolist()
my_f_rm = my_f_rm + ['Unnamed: 0', 'Unnamed: 0.1']
my_f = [f for f in my_f_full if f not in my_f_rm]
mydf = mydata[my_f]
mydf.reset_index(inplace = True)

mydf['FCV_Avg'] = (mydf['3062-0.0'] + mydf['3062-0.1'] + mydf['3062-0.2'])/3
mydf['FEV1_Avg'] = (mydf['3063-0.0'] + mydf['3063-0.1'] + mydf['3063-0.2'])/3
mydf['PEF_Avg'] = (mydf['3064-0.0'] + mydf['3064-0.1'] + mydf['3064-0.2'])/3
mydf['HandGripStrengthAvg'] = (mydf['46-0.0'] + mydf['47-0.0'])/2

mydf['LegFatPercentage'] = (mydf['23115-0.0'] + mydf['23111-0.0'])/2
mydf['LegFatMass'] = (mydf['23112-0.0'] + mydf['23116-0.0'])/2
mydf['ImpedanceOfLegAvg'] = (mydf['23107-0.0'] + mydf['23108-0.0'])/2
mydf['ImpedanceOfArmAvg'] = (mydf['23110-0.0'] + mydf['23111-0.0'])/2
mydf['ImpedanceOfLimbAvg'] = (mydf['ImpedanceOfLegAvg'] + mydf['ImpedanceOfArmAvg'])/2

mydf['NbCorrectMatchesTotal'] = mydf['398-0.1'] + mydf['398-0.2']
mydf['CorrectMatchesPer1'] = mydf['398-0.1']/(mydf['398-0.1'] + mydf['399-0.1'])
mydf['CorrectMatchesPer2'] = mydf['398-0.2']/(mydf['398-0.2'] + mydf['399-0.2'])
mydf['CorrectMatchesPerAvg'] = (mydf['CorrectMatchesPer1'] + mydf['CorrectMatchesPer2'])/2

mydf['NbTimeSnapButtonPressedAvg'] = (mydf['403-0.1'] + mydf['403-0.1'] + mydf['403-0.2'] +
                                      mydf['403-0.3'] + mydf['403-0.4'] + mydf['403-0.5'] +
                                      mydf['403-0.6'] + mydf['403-0.7'] + mydf['403-0.8'] +
                                      mydf['403-0.9'] + mydf['403-0.10'] + mydf['403-0.11'])/12

mydf['Duration1StPressSnapButtonAvg'] = (mydf['404-0.5'] + mydf['404-0.7'] + mydf['404-0.10'] + mydf['404-0.11'])/4

mydf['400-0.1'].loc[mydf['400-0.1'] == 0] = np.nan
mydf['400-0.2'].loc[mydf['400-0.2'] == 0] = np.nan
mydf['Time2CompeleteRoundTotal'] = mydf['400-0.1'] + mydf['400-0.2']
mydf['CorrectMatchesEfficay1'] = mydf['400-0.1']/mydf['398-0.1']
mydf['CorrectMatchesEfficay2'] = mydf['400-0.2']/mydf['398-0.2']
mydf['CorrectMatchesEfficayAvg'] = (mydf['CorrectMatchesEfficay1'] + mydf['CorrectMatchesEfficay2'])/2


mydf.to_csv(dpath + 'Preprocessed_Data/Preprocessed_Data.csv')
