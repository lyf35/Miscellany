
# # coding: utf-8

# # In[1]:


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
import gc
import seaborn as sns
import warnings
warnings.filterwarnings("ignore")

from scipy.signal import hilbert
from scipy.signal import hann
from scipy.signal import convolve
from scipy import stats

from catboost import CatBoostRegressor
from sklearn.ensemble import RandomForestRegressor
from xgboost.sklearn import XGBRegressor
from lightgbm.sklearn import LGBMRegressor
from sklearn.kernel_ridge import KernelRidge
from sklearn.svm import NuSVR
from bayes_opt import BayesianOptimization
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import GridSearchCV


X_train_scaled = pd.read_csv('../input/lanlfeatureextraction/scaled_train_X.csv',header=0,index_col=None)
X_test_scaled = pd.read_csv('../input/lanlfeatureextraction/scaled_test_X.csv',header=0,index_col=None)
y_tr=pd.read_csv('../input/lanlfeatureextraction/train_y.csv',header=0,index_col=None)
submission=pd.read_csv('../input/LANL-Earthquake-Prediction/sample_submission.csv',header=0,index_col=None)


params_xgb={
    'max_depth': (5,10),
    'subsample': (0.2,0.8),
    '_lambda':(0.1,2),
    'alpha':(0.1,2),
    'colsample_bytree':(0.2,0.8),
    'min_child_weight':(0.1,2),
    'gamma':(0.1,2),
}

def xgb_cv_score(max_depth,subsample,_lambda,alpha,colsample_bytree,min_child_weight,gamma):
    params= {
        'max_depth':int(max_depth),
        'subsample':float(subsample),
        'lambda':float(_lambda),
        'alpha':float(alpha),
        'colsample_bytree':float(colsample_bytree),
        'min_child_weight':float(min_child_weight),
        'gamma':float(gamma),
        'eval_metric': 'mae',
        'tree_method':'gpu_hist',
        'silent': True,
        'n_estimators':25000,
        'early_stopping_rounds':500
    }
    val=cross_val_score(XGBRegressor(**params),X_train_scaled,y_tr,scoring='neg_mean_absolute_error',cv=5).mean()
    print(params)
    print('mean MAE: ',val)
    return val

xgb_bo=BayesianOptimization(xgb_cv_score,params_xgb)
xgb_bo.maximize()
max_param_xgb=xgb_bo.max['params']
lambda_value=max_param_xgb['_lambda']
max_param_xgb.pop('_lambda')
max_param_xgb['lambda']=lambda_value
max_param_xgb['eval_metric']='mae'
max_param_xgb['silent']=True
max_param_xgb['n_estimators']=80000
max_param_xgb['early_stopping_rounds']=500
max_param_xgb['tree_method']='gpu_hist'
max_param_xgb['max_depth']=int(max_param_xgb['max_depth'])
print(max_param_xgb)
opt_xgb_reg=XGBRegressor(**max_param_xgb)
opt_xgb_reg.fit(X_train_scaled,y_tr)
y_pred = opt_xgb_reg.predict(X_test_scaled).reshape(-1,1)

# params_xgb_grid={
#     'max_depth': [6,9,12],
#     'subsample': [0.7,0.85,1.0],
#     #'lambda':[0.1,0.5,1],
#     'alpha':[0.1,0.5,1],
#     'min_child_weight':[0.5,1.2,2],
#     'gamma':[0,0.5,1,2]
# }
# xgb_reg=XGBRegressor(eval_metric='mae',tree_method='gpu_hist',silent=True,n_estimators=20000,early_stopping_rounds=200)
# grid_search_xgb=GridSearchCV(xgb_reg,params_xgb_grid,cv=5,scoring='neg_mean_absolute_error')
# grid_search_xgb.fit(X_train_scaled,y_tr)
# print(grid_search_xgb.cv_results_)
# print(grid_search_xgb.best_params_)
# y_pred=grid_search_xgb.predict(X_test_scaled).reshape(-1,)

submission['time_to_failure'] = y_pred
# submission['time_to_failure'] = prediction_lgb_stack
submission.to_csv('xgboost.csv',index=False)

