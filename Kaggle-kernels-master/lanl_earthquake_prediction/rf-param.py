
# coding: utf-8

# In[1]:


import numpy as np
import pandas as pd
import os

import matplotlib.pyplot as plt
from tqdm import tqdm_notebook
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_absolute_error
pd.options.display.precision = 15

import time
import datetime
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import StratifiedKFold, KFold, RepeatedKFold
from sklearn.metrics import mean_absolute_error
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
import gc
import seaborn as sns
import warnings
warnings.filterwarnings("ignore")

from scipy.signal import hilbert
from scipy.signal import hann
from scipy.signal import convolve
from scipy import stats
import tensorflow as tf

from catboost import CatBoostRegressor
from xgboost.sklearn import XGBRegressor
from lightgbm.sklearn import LGBMRegressor
from sklearn.kernel_ridge import KernelRidge
from sklearn.svm import NuSVR
from bayes_opt import BayesianOptimization
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import train_test_split
from sklearn.model_selection import GridSearchCV


X_train_scaled = pd.read_csv('../input/lanlfeatureextraction/scaled_train_X.csv',header=0,index_col=None)
X_test_scaled = pd.read_csv('../input/lanlfeatureextraction/scaled_test_X.csv',header=0,index_col=None)
y_tr=pd.read_csv('../input/lanlfeatureextraction/train_y.csv',header=0,index_col=None)
submission=pd.read_csv('../input/LANL-Earthquake-Prediction/sample_submission.csv',header=0,index_col=None)

# params_rf={
#     'max_depth': (5,15),
#     'min_samples_leaf':(1,500),
#     'min_samples_split':(30,600),
# }

# def rf_cv_score(max_depth,min_samples_leaf,min_samples_split):
    
#     params= {
#         'max_depth':int(max_depth),
#         'min_samples_leaf': int(min_samples_leaf),
#         'min_samples_split':int(min_samples_split),
#         'n_estimators':5000,
#         'bootstrap':True
#     }
#     train_set_x,test_set_x,train_set_y,test_set_y=train_test_split(X_train_scaled,y_tr,test_size=0.3)
#     learner=RandomForestRegressor(**params)
#     learner.fit(train_set_x,train_set_y)
#     pred_y=learner.predict(test_set_x)
#     pred_y=pred_y.reshape(-1,1)
#     val=abs(pred_y-test_set_y).mean(axis=0)
#     #val=cross_val_score(RandomForestRegressor(**params),X_train_scaled,y_tr,scoring='neg_mean_absolute_error',cv=5).mean()
#     return val['time_to_failure']

# rf_bo=BayesianOptimization(rf_cv_score,params_rf)
# rf_bo.maximize()
# max_param_rf=rf_bo.max['params']
# max_param_rf['max_depth']=int(max_param_rf['max_depth'])
# max_param_rf['min_samples_leaf']=int(max_param_rf['min_samples_leaf'])
# max_param_rf['min_samples_split']=int(max_param_rf['min_samples_split'])
# max_param_rf['n_estimators']=30000
# max_param_rf['bootstrap']=True
# print(max_param_rf)
# opt_rf_reg=RandomForestRegressor(**max_param_rf)
opt_rf_reg=RandomForestRegressor(n_estimators=1500,bootstrap=True,max_depth=12)
opt_rf_reg.fit(X_train_scaled,y_tr)
y_pred = opt_rf_reg.predict(X_test_scaled).reshape(-1,)

# params_rf_grid={
#     'max_depth': [6,9,12],
#     'min_samples_leaf':[1,200,500],
#     'min_samples_split':[20,50,100,150],
# }
# rf_reg=RandomForestRegressor(bootstrap=True,n_estimators=5000)
# grid_search_rf=GridSearchCV(rf_reg,params_rf_grid,cv=5,scoring='neg_mean_absolute_error')
# grid_search_rf.fit(X_train_scaled,y_tr)
# print(grid_search_rf.cv_results_)
# y_pred=grid_search_rf.predict(X_test_scaled).reshape(-1,)
submission['time_to_failure'] = y_pred
submission.to_csv('randomforest.csv',index=False)

