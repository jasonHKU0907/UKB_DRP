

import numpy as np
import pandas as pd
import sklearn
import scipy

dpath = '/Volumes/JasonWork/UKB/'
stroke_df = pd.read_csv(dpath + 'Data/stroke.csv', usecols = ['eid', '42006-0.0'])
target_df = pd.read_csv(dpath + 'Data/dementia.csv')
mydf = pd.merge(target_df, stroke_df, how='inner', on=['eid'])

def get_days_intervel(start_date_var, end_date_var, df):
    start_date = pd.to_datetime(df[start_date_var])
    end_date = pd.to_datetime(df[end_date_var])
    nb_of_dates = start_date.shape[0]
    days = [(end_date[i] - start_date[i]).days for i in range(nb_of_dates)]
    months = [ele/30 for ele in days]
    years = [ele/365 for ele in days]
    return pd.DataFrame({'stroke_days':days, 'stroke_months':months, 'stroke_years':years})

tmp = get_days_intervel('Re_date.x', '42006-0.0', mydf)
mydf_stroke = pd.concat((mydf, tmp), axis = 1)
mydf_stroke['stroke_status'] = 1
mydf_stroke['stroke_status'][mydf_stroke['stroke_years'].isnull()==True] = 0

mydf_stroke.to_csv(dpath + 'Preprocessed_Data/target.csv')



