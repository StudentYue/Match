# -*- coding: utf-8 -*-
"""
Created on Sat Jan 19 12:10:15 2019

@author: ZHOU-JC
"""

# -*- coding: utf-8 -*-
"""
Created on Fri Nov 30 16:19:26 2018

@author: ZHOU-JC
"""

import pandas as pd 
import numpy  as np
from scipy.stats import pearsonr
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error
from sklearn.feature_selection import RFE
from sklearn.tree import DecisionTreeRegressor
from sklearn.preprocessing import StandardScaler, MinMaxScaler, LabelEncoder
from sklearn.linear_model import Lasso,LassoCV,LassoLarsCV   # Lasso回归,LassoCV交叉验证实现alpha的选取，LassoLarsCV基于最小角回归交叉验证实现alpha的选取
from sklearn.linear_model import RidgeCV
from sklearn.model_selection import train_test_split, GridSearchCV

train_data  = pd.read_csv(r'C:\Users\ZHOU-JC\Desktop\数据挖掘项目\PUBG Finish Placement Prediction(Kernels Only)\train_V2.csv', nrows=500000)
test_data   = pd.read_csv(r'C:\Users\ZHOU-JC\Desktop\数据挖掘项目\PUBG Finish Placement Prediction(Kernels Only)\test_V2.csv', nrow=100000)
