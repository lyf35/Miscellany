#The training of Catboost is too slow. So use the parameter of Catboost from Andrew directly. In his kernel, only loss=MAE is used in the parameter list.
import numpy as np
import pandas as pd
import os

from sklearn.model_selection import train_test_split
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


params_cat = {
    'depth':(5,10),
    'l2_leaf_reg':(0.01,5),
    'min_data_in_leaf':(1,100),
}

def cat_cv_score(depth,min_data_in_leaf,l2_leaf_reg):
    params= {
        'depth':int(depth),
        'min_data_in_leaf':int(min_data_in_leaf),
        'l2_leaf_reg':float(l2_leaf_reg),
        'early_stopping_rounds':200,
        'iterations':10000,
        'loss_function':'MAE',
        'task_type':'GPU',
        'max_bin':48,
        'verbose':10000,
        'allow_writing_files':False,
        'boosting_type':'Plain',
    }
    val=cross_val_score(CatBoostRegressor(**params),X_train_scaled,y_tr,scoring='neg_mean_absolute_error',cv=5).mean()
    print(params)
    print('MAE:',val)
    return val

cat_bo=BayesianOptimization(cat_cv_score,params_cat)
cat_bo.maximize()
max_param_cat=cat_bo.max['params']
max_param_cat['early_stopping_rounds']=200
max_param_cat['depth']=int(max_param_cat['depth'])
max_param_cat['min_data_in_leaf']=int(max_param_cat['min_data_in_leaf'])
max_param_cat['iterations']=50000
max_param_cat['loss_function']='MAE'
max_param_cat['verbose']=10000
max_param_cat['allow_writing_files']=False
max_param_cat['task_type']='GPU'
max_param_cat['learning_rate']=0.01
max_param_cat['boosting_type']='Plain'
print(max_param_cat)
opt_cat_reg=CatBoostRegressor(**max_param_cat)
opt_cat_reg.fit(X_train_scaled,y_tr)
y_pred = opt_cat_reg.predict(X_test_scaled).reshape(-1,)


# params_cat_grid={
#     'depth':[6,8,10,12],
#     'l2_leaf_reg':[0,0.3,0.6,0.9,1.2,1.5],
# }
# cat_reg=CatBoostRegressor(iterations=5000,loss_function='MAE',verbose=50000,task_type='GPU',allow_writing_files=False)
# grid_search_cat=GridSearchCV(cat_reg,params_cat_grid,cv=5,scoring='neg_mean_absolute_error')
# grid_search_cat.fit(X_train_scaled,y_tr)
# print(grid_search_cat.cv_results_)
# y_pred=grid_search_cat.predict(X_test_scaled).reshape(-1,)


submission['time_to_failure'] = y_pred
# submission['time_to_failure'] = prediction_lgb_stack
submission.to_csv('catboost.csv',index=False)