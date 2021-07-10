
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


# params_nusvr={
#     'nu':(0.001,1),
#     'C':(0.01,15),
#     'tol':(0.001,1.5)
# }

# def nusvr_cv_score(nu,C,tol):
#     params= {
#         'nu':float(nu),
#         'C':float(C),
#         'tol':float(tol),
#         'gamma':'auto'
#     }
#     val=cross_val_score(NuSVR(**params),X_train_scaled,y_tr,scoring='neg_mean_absolute_error',cv=5).mean()
#     return val

# nusvr_bo=BayesianOptimization(nusvr_cv_score,params_nusvr)
# nusvr_bo.maximize()
# max_param_nusvr=nusvr_bo.max['params']
# max_param_nusvr['gamma']='auto'
# print(max_param_nusvr)
# opt_nusvr_reg=NuSVR(**max_param_nusvr)

# NuSVR(gamma='scale', nu=0.7, tol=0.01, C=1.0)
opt_nusvr_reg=NuSVR(gamma='scale', nu=0.9, C=10.0, tol=0.01)
opt_nusvr_reg.fit(X_train_scaled,y_tr)
y_pred = opt_nusvr_reg.predict(X_test_scaled).reshape(-1,)

# params_nusvr_grid={
#     'nu':[0.1,0.3,0.5,0.7,1],
#     'C':[1],
#     'tol':[0.01,0.03,0.05,0.07,0.1]
# }
# nusvr_reg=NuSVR()
# grid_search_nusvr=GridSearchCV(nusvr_reg,params_nusvr_grid,cv=4,scoring='neg_mean_absolute_error')
# grid_search_nusvr.fit(X_train_scaled,y_tr)
# print(grid_search_nusvr.cv_results_)
# y_pred=grid_search_nusvr.predict(X_test_scaled).reshape(-1,)

submission['time_to_failure'] = y_pred
# submission['time_to_failure'] = prediction_lgb_stack
submission.to_csv('nusvr.csv',index=False)

