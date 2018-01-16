# -*- coding: utf-8 -*-
"""
Created on Sun Jan  7 16:52:56 2018

@author: ATruong1
"""

# -*- coding: utf-8 -*-
"""
Created on Wed Nov 22 13:29:25 2017

@author: ATruong1
"""
import pandas as pd
import numpy as np
from sklearn.preprocessing import OneHotEncoder
from sklearn.cross_validation import train_test_split
from scipy import sparse # Need this to create a sparse array
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import SGDRegressor
from sklearn.grid_search import GridSearchCV
from sklearn.metrics import mean_absolute_error, mean_squared_error
import datetime
from datetime import datetime, date, timedelta
from dateutil.parser import parse
from sklearn.externals import joblib
#%%% -------- concaténation des variables ------------
#On va concaténer les matrices
months = ['01', '02', '03', '04', '05', '06', '07', '08', '09', '10', '11', '12']
lis_var= ['MONTH', 'DAY_OF_MONTH', 'DAY_OF_WEEK', 'UNIQUE_CARRIER','FL_NUM','ORIGIN', 'DEST', 'CRS_DEP_TIME', 
          'CRS_ARR_TIME', 'ARR_DELAY','DEP_DELAY', 'DISTANCE', 'CRS_ELAPSED_TIME', 'ACTUAL_ELAPSED_TIME',
          'DIVERTED', 'CANCELLED','CARRIER_DELAY','WEATHER_DELAY','NAS_DELAY','SECURITY_DELAY',
          'LATE_AIRCRAFT_DELAY','FL_DATE']
var_ref =['ORIGIN_AIRPORT_ID','ORIGIN','ORIGIN_CITY_NAME']
data_=[]
data_ref=[]
for i in months:
    print('début '+i)
    data_int = pd.read_csv('2016_'+i+'.csv', sep=",",error_bad_lines=False)
    data_int_ = data_int[data_int.columns.intersection(lis_var)]
    ref_int = data_int[data_int.columns.intersection(var_ref)].drop_duplicates()
    data_.append(data_int_)
    data_ref.append(ref_int)
    print('fin '+i)
data_ = pd.concat(data_)#5 635 978 flights
data_ref = pd.concat(data_ref)
data_ref = data_ref.drop_duplicates('ORIGIN')

data_ref = data_ref[data_ref['ORIGIN_CITY_NAME'].str.isnumeric()==False]
data_ref = pd.concat([data_ref, data_ref['ORIGIN_CITY_NAME'].apply(lambda x: pd.Series([i for i in x.split(',')]))], axis = 1)
data_ref.rename(columns={0:'CITY', 1:'STATE', 'ORIGIN_AIRPORT_ID':'AIRPORT_ID', 'ORIGIN':'CODE'},inplace=True)
data_ref = data_ref.drop('ORIGIN_CITY_NAME', axis = 1)#313 airports
data_ = data_.drop_duplicates()# 5 528 192 flights
#%%--------------- Suppression des retardds imprévisibles -------
data_v2 = data_
data_v2 = data_v2[(data_v2['CANCELLED'] == 0) & (data_v2['DIVERTED'] == 0)] # 5 449 337
data_v2 = data_v2[(data_v2['ARR_DELAY']>=-35) & (data_v2['ARR_DELAY']<=66)]
#%%
#Heure HH:MM => HH
data_v2['DEP_HOUR'] = data_v2['CRS_DEP_TIME'].divide(100).astype('int')
data_v2['ARR_HOUR'] = data_v2['CRS_ARR_TIME'].divide(100).astype('int')
data_v2['DEP_HOUR'] = data_v2['DEP_HOUR'].replace(24,0)
data_v2['ARR_HOUR'] = data_v2['ARR_HOUR'].replace(24,0)
data_v2['FL_DATE'] = data_v2['FL_DATE'].astype('str')
data_v2['WEEK'] = data_v2['FL_DATE'].apply(lambda x: datetime(2016, int(x.split('-')[1]), 
                                           int(x.split('-')[2])).isocalendar()[1])
#%% ---------------------------- Modèle 2 - sans distance et elapsed time -----------------------------------------
data_M2 = data_v2

#%%
data_M2['UNIQUE_CARRIER'] = data_M2['UNIQUE_CARRIER'].astype('category')
data_M2['CARRIER_CODE'] = data_M2['UNIQUE_CARRIER'].cat.codes
data_carrier_M2 = data_M2[['CARRIER_CODE','UNIQUE_CARRIER']].drop_duplicates()
data_M2['ORIGIN_NUM'] = data_M2['ORIGIN'].astype('category')
data_M2['ORIGIN_NUM'] = data_M2['ORIGIN_NUM'].cat.codes
data_airport_M2 = data_M2[['ORIGIN','ORIGIN_NUM']].drop_duplicates()
#%%
data_airport_M2 = data_airport_M2.set_index('ORIGIN')
data_carrier_M2 = data_carrier_M2.set_index('UNIQUE_CARRIER')
dict_airport_M2 = data_airport_M2.to_dict()
data_M2['DEST_NUM'] = data_M2['DEST'].apply(lambda x: dict_airport_M2['ORIGIN_NUM'][x])
#%% ------- rajout des vacances ---------------
list_holidays = ['2016-01-01', '2016-01-18', '2016-02-15', '2016-05-30', '2016-07-04', '2016-09-05',
                 '2016-10-10','2016-11-11', '2016-11-24', '2016-12-26', '2017-01-02']
list_holidays_ = [parse(x) for x in list_holidays]
d1 = datetime(2016, 1, 2)  # start date
d2 = datetime(2016, 12, 31)  # end date
delta = d2 - d1
date1 = datetime(2016, 1, 1)
dist_holidays = pd.DataFrame([(date1.strftime('%Y-%m-%d'),1/(min([abs((date1-y).days) for y in list_holidays_])+1))],columns=['FL_DATE','HDAYS'])
for i in range(delta.days+1):
    dat = d1 + timedelta(days=i)
    day = pd.DataFrame([(dat.strftime('%Y-%m-%d'), 1/(min([abs((dat-y).days) for y in list_holidays_])+1))],columns=['FL_DATE','HDAYS'])
    dist_holidays = dist_holidays.append(day)
dist_holidays = dist_holidays.set_index('FL_DATE')
dict_holidays = dist_holidays.to_dict()
data_M2['HDAYS'] = data_M2['FL_DATE'].apply(lambda x: dict_holidays['HDAYS'][x])
#%%
data_M2['FLIGHT'] = data_M2['ORIGIN'].astype('str') + '|' + data_M2['DEST'].astype('str') + '|' + data_M2['UNIQUE_CARRIER'].astype('str') 
#%%----------------Modèle 6-----------------------
data_M6 = data_M2
scalingDF_M6 = data_M6[['DISTANCE', 'CRS_ELAPSED_TIME','HDAYS']].astype('float') # Numerical features
categDF_M6 = data_M6[['MONTH', 'DAY_OF_WEEK', 'ORIGIN_NUM', 
                    'DEST_NUM', 'ARR_HOUR', 'DEP_HOUR', 
                    'CARRIER_CODE','WEEK']]
#%%% ------------ Prédiction linéaire all -------
encoder = OneHotEncoder() # Create encoder objec
categDF_encoded = encoder.fit_transform(categDF_M6)
scalingDF_sparse = sparse.csr_matrix(scalingDF_M6) # Transform the data and convert to sparse
x_final = sparse.hstack((scalingDF_sparse, categDF_encoded))
y_final = data_M6['ARR_DELAY'].values
x_train, x_test, y_train, y_test = train_test_split(x_final,y_final,test_size = 0.2,random_state = 0) # Do 80/20 split
size_scale = scalingDF_M6.shape[1]
x_train_numerical = x_train[:, 0:size_scale].toarray() # We only want the first two features which are the numerical ones.
x_test_numerical = x_test[:, 0:size_scale].toarray()
scaler = StandardScaler() # create scaler object
scaler.fit(x_train_numerical) # fit with the training data ONLY
x_train_numerical = sparse.csr_matrix(scaler.transform(x_train_numerical)) # Transform the data and convert to sparse
x_test_numerical = sparse.csr_matrix(scaler.transform(x_test_numerical))
x_train[:, 0:size_scale] = x_train_numerical
x_test[:, 0:size_scale] = x_test_numerical
SGD_params = [{'loss':['huber','epsilon_insensitive','squared_epsilon_insensitive'], 'alpha': 10.0**-np.arange(1,7),'l1_ratio' : [0,.5,1], 'epsilon' : [1,5,10]}] # Suggested range we try
SGD_model_M6 = GridSearchCV(SGDRegressor(penalty = 'elasticnet', random_state = 0), SGD_params, scoring = 'neg_mean_absolute_error', cv = 5) # Use 5-fold CV 
SGD_model_M6.fit(x_train, y_train) # Fit the model
best_params = SGD_model_M6.best_params_ #alpha = 10^-6 / penalty = l1_ratio = 1
y_true, y_pred = y_test, SGD_model_M6.predict(x_test) # Predict on our test set
MAE_M6 = mean_absolute_error(y_true, y_pred)
MSE_M6 = mean_squared_error(y_true, y_pred)
print('Mean absolute error of SGD regression was:')
print(MAE_M6) ##12,42 0.00001 1 / std 16.9 / 285.4
print(MSE_M6) ##285.4
#%%
coefs = SGD_model_M6.best_estimator_.coef_
intercept = SGD_model_M6.best_estimator_.intercept_
#%%
list_coefs=pd.DataFrame(np.insert(coefs, 0, intercept[0]), columns =['INTERCEPT_COEFS'])
#%%
list_coefs.to_csv(path_or_buf = 'coefs_global_.csv', sep=',')
#%%
joblib.dump(encoder, 'encoding.pkl')
joblib.dump(scaler, 'scaling.pkl')
#%%
list_flight = data_M6[['FLIGHT','DISTANCE', 'CRS_ELAPSED_TIME']].drop_duplicates('FLIGHT')
#%%
list_flight.to_csv(path_or_buf = 'list_flight.csv', sep=',')


