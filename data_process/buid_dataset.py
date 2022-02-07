import pandas as pd
import numpy as np

df = pd.read_csv('MCIRD_aaic2021_train.csv')
#
# data = pd.isnull(df['total_call_duration'])
# print(df['total_call_duration'][data])



MLPcols = ['subscriber_gender','subscriber_age','is_usage_nonzero','is_data_usage_nonzero','#activated_monthly_data_packages','#activated_short_term_data_packages','#activated_type_one_data_packages','#activated_type_two_data_packages','#activated_type_three_data_packages']
MLP_features = df[MLPcols]


RNNcols = ['subscriber_total_expenses','nonpackage_voice_expenses',
       'total_call_duration','data_cash_expenses',
       'nonpackage_data_expenses', 'package_data_noncash_expenses',
       'subscriber_data_expenses', 'subscriber_nondata_expenses',
       'data_usage_volume']

RNN_features = df[RNNcols].copy()

for i in RNNcols:
       data = pd.isnull(RNN_features[i])
       RNN_features[i][data] = np.mean(RNN_features[i])

RNN_features['total_call_duration'][data] = np.mean(RNN_features['total_call_duration'])
print(RNN_features['total_call_duration'][data])


RNN_features.to_csv(r'/home/saeed/timeseries/datasets/financial/mcird.txt', header=None, index=None, sep=',', mode='a')
