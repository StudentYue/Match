# -*- coding: utf-8 -*-
"""
Created on Sat Nov 24 15:15:06 2018

@author: ZHOU-JC
"""
import pandas as pd
import numpy as np
import seaborn as sns
from scipy import stats
from scipy.stats import skew
from scipy.stats import norm
import matplotlib
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.manifold import TSNE
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler

train_df = pd.read_csv(r"C:\Users\ZHOU-JC\Desktop\kaggle\House Prices Advanced Regression Techniques/train.csv")
test_df = pd.read_csv(r"C:\Users\ZHOU-JC\Desktop\kaggle\House Prices Advanced Regression Techniques/test.csv")

all_df = pd.concat((train_df.loc[:,'MSSubClass':'SaleCondition'], test_df.loc[:,'MSSubClass':'SaleCondition']), axis=0,ignore_index=True)
all_df['MSSubClass'] = all_df['MSSubClass'].astype(str)
quantitative = [f for f in all_df.columns if all_df.dtypes[f] != 'object']
qualitative = [f for f in all_df.columns if all_df.dtypes[f] == 'object']

#%% 确实数据处理
missing = all_df.isnull().sum()
missing.sort_values(inplace=True,ascending=False)
missing = missing[missing > 0]
#dealing with missing data
all_df = all_df.drop(missing[missing>1].index,1)
# 对于missing 1 的我们到时候以平均数填充
all_df.isnull().sum()[all_df.isnull().sum()>0]


#%%处理log项
logfeatures = ['GrLivArea','1stFlrSF','2ndFlrSF','TotalBsmtSF','LotArea','KitchenAbvGr','GarageArea']
for logfeature in logfeatures:
    all_df[logfeature] = np.log1p(all_df[logfeature].values)


#%%处理Boolean变量
all_df['HasBasement'] = all_df['TotalBsmtSF'].apply(lambda x: 1 if x > 0 else 0)
all_df['HasGarage'] = all_df['GarageArea'].apply(lambda x: 1 if x > 0 else 0)
all_df['Has2ndFloor'] = all_df['2ndFlrSF'].apply(lambda x: 1 if x > 0 else 0)
all_df['HasWoodDeck'] = all_df['WoodDeckSF'].apply(lambda x: 1 if x > 0 else 0)
all_df['HasPorch'] = all_df['OpenPorchSF'].apply(lambda x: 1 if x > 0 else 0)
all_df['HasPool'] = all_df['PoolArea'].apply(lambda x: 1 if x > 0 else 0)
all_df['IsNew'] = all_df['YearBuilt'].apply(lambda x: 1 if x > 2000 else 0)
quantitative = [f for f in all_df.columns if all_df.dtypes[f] != 'object']
qualitative = [f for f in all_df.columns if all_df.dtypes[f] == 'object']

quantitative = [f for f in all_df.columns if all_df.dtypes[f] != 'object']
qualitative = [f for f in all_df.columns if all_df.dtypes[f] == 'object']

#%%对于定性变量的encode
all_dummy_df = pd.get_dummies(all_df)


#%%对于数值变量标准化
mean_cols = all_dummy_df.mean()
all_dummy_df = all_dummy_df.fillna(mean_cols)

X = all_dummy_df[quantitative]
std = StandardScaler()
s = std.fit_transform(X)

all_dummy_df[quantitative] = s


dummy_train_df = all_dummy_df.loc[train_df.index]
dummy_test_df = all_dummy_df.loc[test_df.index]
#
y_train = np.log(train_df.SalePrice)

from sklearn.linear_model import Ridge
from sklearn.model_selection import cross_val_score

def rmse_cv(model):
    rmse= np.sqrt(-cross_val_score(model, dummy_train_df, y_train.values, scoring="neg_mean_squared_error", cv = 5))
    return(rmse)

#%% 岭回归
alphas = np.logspace(-3, 2, 50)
cv_ridge = []
coefs = []
for alpha in alphas:
    model = Ridge(alpha = alpha)
    model.fit(dummy_train_df,y_train)
    cv_ridge.append(rmse_cv(model).mean())
    coefs.append(model.coef_)
import matplotlib.pyplot as plt

cv_ridge = pd.Series(cv_ridge, index = alphas)
cv_ridge.plot(title = "Validation - Just Do It")
plt.xlabel("alpha")
plt.ylabel("rmse")
# plt.plot(alphas, cv_ridge)
# plt.title("Alpha vs CV Error")

# 岭迹图
# matplotlib.rcParams['figure.figsize'] = (12.0, 12.0)
ax = plt.gca()

# ax.set_color_cycle(['b', 'r', 'g', 'c', 'k', 'y', 'm'])

ax.plot(alphas, coefs)
ax.set_xscale('log')
ax.set_xlim(ax.get_xlim()[::-1])  # reverse axis
plt.xlabel('alpha')
plt.ylabel('weights')
plt.title('Ridge coefficients as a function of the regularization')
plt.axis('tight')
plt.show()

#%% Lasso
from sklearn.linear_model import Lasso,LassoCV
# alphas = np.logspace(-3, 2, 50)
# alphas = [1, 0.1, 0.001, 0.0005]
alphas = np.logspace(-4, -2, 100)
cv_lasso = []
coefs = []
for alpha in alphas:
    model = Lasso(alpha = alpha,max_iter=5000)
    model.fit(dummy_train_df,y_train)
    cv_lasso.append(rmse_cv(model).mean())
    coefs.append(model.coef_)
cv_lasso = pd.Series(cv_lasso, index = alphas)
cv_lasso.plot(title = "Validation - Just Do It")
plt.xlabel("alpha")
plt.ylabel("rmse")

model = Lasso(alpha = 0.00058,max_iter=5000)
model.fit(dummy_train_df,y_train)

coef = pd.Series(model.coef_, index = dummy_train_df.columns)
print("Lasso picked " + str(sum(coef != 0)) + " variables and eliminated the other " +  str(sum(coef == 0)) + " variables")

matplotlib.rcParams['figure.figsize'] = (8.0, 10.0)
imp_coef.plot(kind = "barh")
plt.title("Coefficients in the Lasso Model")


#%% Elastic Net
from sklearn.linear_model import ElasticNet,ElasticNetCV
elastic = ElasticNetCV(l1_ratio=[.1, .5, .7, .9, .95, .99, 1], 
                                    alphas=[0.001, 0.05, 0.1, 0.3, 1, 3, 5, 10, 15, 30, 50, 75], cv=5,max_iter=5000)
elastic.fit(dummy_train_df, y_train)


#%% XGBoost
import xgboost as xgb

model_xgb = xgb.XGBRegressor(
                 colsample_bytree=0.2,
                 gamma=0.0,
                 learning_rate=0.05,
                 max_depth=6,
                 min_child_weight=1.5,
                 n_estimators=7200,                                                                  
                 reg_alpha=0.9,
                 reg_lambda=0.6,
                 subsample=0.2,
                 seed=42,
                 silent=1)

model_xgb.fit(train_df_munged, label_df.SalePrice)

# Run prediction on training set to get a rough idea of how well it does.
pred_Y_xgb = model_xgb.predict(train_df_munged)























