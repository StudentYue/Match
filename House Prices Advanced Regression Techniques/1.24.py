# -*- coding: utf-8 -*-

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import pandas as pd 
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import KFold, cross_val_score	
from sklearn.model_selection import GridSearchCV
import math
from sklearn.preprocessing import LabelEncoder
from scipy import stats
from scipy.stats import norm, skew #for some statistics
import matplotlib.pyplot as plt
#%%
n_splits = 5
def rmsle_cv(model, X, Y):
    kf = KFold(n_splits=5, shuffle=True, random_state=4).get_n_splits(X.values)
    rmse = (-cross_val_score(model, X, Y, scoring="neg_mean_squared_error", cv = kf))
    return (np.sqrt(rmse.sum()/n_splits))/Y.mean()

# The error metric: RMSE on the log of the sale prices.
from sklearn.metrics import mean_squared_error
def rmse(y_true, y_pred):
    return np.sqrt(mean_squared_error(y_true, y_pred))

#%%
train_data = pd.read_csv(r"C:\Users\ZHOU-JC\Desktop\data match\House Prices Advanced Regression Techniques/train.csv")
test_data  = pd.read_csv(r"C:\Users\ZHOU-JC\Desktop\data match\House Prices Advanced Regression Techniques/test.csv")

#%% 对目标值SalePrice进行正态变换
#sns.distplot(train_data['SalePrice'] , fit=norm);
#(mu, sigma) = norm.fit(train_data['SalePrice'])
#print( '\n mu = {:.2f} and sigma = {:.2f}\n'.format(mu, sigma))
#
#fig = plt.figure()
#res = stats.probplot(train_data['SalePrice'], plot=plt)
#plt.show()

#%% 部分算法喜欢正态分布的数据
train_data["SalePrice"] = np.log1p(train_data["SalePrice"])
sns.distplot(train_data['SalePrice'] , fit=norm);
(mu, sigma) = norm.fit(train_data['SalePrice'])
print( '\n mu = {:.2f} and sigma = {:.2f}\n'.format(mu, sigma))

fig = plt.figure()
res = stats.probplot(train_data['SalePrice'], plot=plt)
plt.show()


#%% 特殊
test_data.loc[666, "GarageQual"] = "TA"
test_data.loc[666, "GarageCond"] = "TA"
test_data.loc[666, "GarageFinish"] = "Unf"
test_data.loc[666, "GarageYrBlt"] = 1980
test_data.loc[1116, "GarageType"] = np.nan

#%%
all_data = train_data.append(test_data, sort=False).reset_index()

#%% 删除一场值
all_data = all_data.drop(all_data[all_data['Id'] == 1299].index)
all_data = all_data.drop(all_data[all_data['Id'] == 524 ].index)
# LotArea异常
all_data = all_data.drop(all_data[all_data['Id'] == 250 ].index)
all_data = all_data.drop(all_data[all_data['Id'] == 314 ].index)
all_data = all_data.drop(all_data[all_data['Id'] == 336 ].index)
all_data = all_data.drop(all_data[all_data['Id'] == 707 ].index)
# LotFrontage异常
all_data = all_data.drop(all_data[all_data['Id'] == 935 ].index)
# GarageArea异常
all_data = all_data.drop(all_data[all_data['Id'] == 582 ].index)
all_data = all_data.drop(all_data[all_data['Id'] == 1191 ].index)

#%% 缺失值处理
total = all_data.isnull().sum().sort_values(ascending=False)
percent = (all_data.isnull().sum())/all_data.isnull().count().sort_values(ascending=False)
missing_data = pd.concat([total, percent], axis=1, sort=False, keys=['Total', 'Percent'])
missing_data[missing_data['Percent'] > 0]
# PoolQc 泳池面积
all_data["PoolQC"] = all_data["PoolQC"].fillna("None")
# MiscFeature 其他特征
all_data["MiscFeature"] = all_data["MiscFeature"].fillna("None")
# Alley 巷子类别
all_data["Alley"] = all_data["Alley"].fillna("None")
# Fence 围墙质量
all_data["Fence"] = all_data["Fence"].fillna("None")
# FireplaceQu 壁炉质量 Fireplaces壁炉数量为0的话 也就没有壁炉了
all_data["FireplaceQu"] = all_data["FireplaceQu"].fillna("None")
# LotFrontage 距离街道的直线距离 
all_data["LotFrontage"] = all_data.groupby("Neighborhood")["LotFrontage"].transform(
    lambda x: x.fillna(x.median()))# GarageQual 车库质量
# GarageYrBlt 车库建造年份
# GarageFinish 车库内饰
# GarageCond 车库条件
# GarageQual 车库质量
# GarageType 车库类型 一般 删除吧
for col in ('GarageCond', 'GarageFinish', 'GarageQual', 'GarageCond'):
    all_data[col] = all_data[col].fillna('None')
#all_data.drop('GarageType', axis=1, inplace=True)
# GarageYrBlt 车库建造年份 这样处理不会有问题？
# GarageArea 车库面积
# GarageCars 车库车容量大小
for col in ('GarageYrBlt', 'GarageArea', 'GarageCars'):
    all_data[col] = all_data[col].fillna(0)
# BsmtExposure 花园地下室墙 删除
# BsmtCond 地下室概况
# BsmtQual 地下室高度
# BsmtFinType2 地下室装饰质量
# BsmtFinType1 地下室装饰质量
for col in ('BsmtQual', 'BsmtCond', 'BsmtExposure', 'BsmtFinType1', 'BsmtFinType2'):
    all_data[col] = all_data[col].fillna('None')    
# MasVnrType 砌体饰面类型
# MasVnrArea 砌体饰面面积
all_data["MasVnrType"] = all_data["MasVnrType"].fillna("None")
all_data["MasVnrArea"] = all_data["MasVnrArea"].fillna(0)
# MSZoning 区域分类
all_data['MSZoning'] = all_data['MSZoning'].fillna(all_data['MSZoning'].mode()[0])
# Utilities 公共设施类型
all_data.drop('Utilities', axis=1, inplace=True)
# Functional 房屋功能性评级
all_data["Functional"] = all_data["Functional"].fillna("Typ")
# Exterior2nd 住宅外墙 #类别很多  这确定可以吗  
# Exterior1st 住宅外墙
all_data['Exterior1st'] = all_data['Exterior1st'].fillna(all_data['Exterior1st'].mode()[0])
all_data['Exterior2nd'] = all_data['Exterior2nd'].fillna(all_data['Exterior2nd'].mode()[0])
# SaleType 交易类型
all_data['SaleType'] = all_data['SaleType'].fillna(all_data['SaleType'].mode()[0])
# BsmtFinSF1 地下室装饰面积
# BsmtFinSF2 地下室装饰面积 
# TotalBsmtSF 地下室总面积
# BsmtUnfSF 地下室未装饰面积
# BsmtHalfBath 地下室半浴室
# BsmtFullBath 地下室全浴室
for col in ('BsmtFinSF1', 'BsmtFinSF2', 'BsmtUnfSF','TotalBsmtSF', 'BsmtFullBath', 'BsmtHalfBath'):
    all_data[col] = all_data[col].fillna(0)
# Electrical 电力系统
all_data['Electrical'] = all_data['Electrical'].fillna(all_data['Electrical'].mode()[0])
# KitchenQual 厨房质量
all_data['KitchenQual'] = all_data['KitchenQual'].fillna(all_data['KitchenQual'].mode()[0])



#%% 对变量进行编码
Qual_map = (lambda x:5 if x == 'Ex' else 4 if x == 'Gd' else 3 if x =='TA'
            else 2 if x =='Fa' else 1 if x=='Po' else 0)
# 序数型参数
all_data['FireplaceQu'] = all_data['FireplaceQu'].map(Qual_map).astype(int)
all_data['BsmtQual'] = all_data['BsmtQual'].map(Qual_map).astype(int)
all_data['BsmtCond'] = all_data['BsmtCond'].map(Qual_map).astype(int)
all_data['GarageQual'] = all_data['GarageQual'].map(Qual_map).astype(int)
all_data['GarageCond'] = all_data['GarageCond'].map(Qual_map).astype(int)
all_data['ExterQual'] = all_data['ExterQual'].map(Qual_map).astype(int)
all_data['ExterCond'] = all_data['ExterCond'].map(Qual_map).astype(int)
all_data['HeatingQC'] = all_data['HeatingQC'].map(Qual_map).astype(int)
all_data['PoolQC'] = all_data['PoolQC'].map(Qual_map).astype(int)  #大部分确实
all_data['KitchenQual'] = all_data['KitchenQual'].map(Qual_map).astype(int)  #大部分确实
all_data['OverallCond'] = all_data['OverallCond'].astype(int).astype(int)

all_data["BsmtExposure"] = all_data["BsmtExposure"].map({'None': 0, "No": 1, "Mn": 2, "Av": 3, "Gd": 4}).astype(int)

bsmt_fin_dict = {'None': 0, "Unf": 1, "LwQ": 2, "Rec": 3, "BLQ": 4, "ALQ": 5, "GLQ": 6}
all_data["BsmtFinType1"] = all_data["BsmtFinType1"].map(bsmt_fin_dict).astype(int)
all_data["BsmtFinType2"] = all_data["BsmtFinType2"].map(bsmt_fin_dict).astype(int)

all_data["Functional"] = all_data["Functional"].map(
        {'None': 0, "Sal": 1, "Sev": 2, "Maj2": 3, "Maj1": 4, 
         "Mod": 5, "Min2": 6, "Min1": 7, "Typ": 8}).astype(int)
    
all_data["GarageFinish"] = all_data["GarageFinish"].map(
        {'None': 0, "Unf": 1, "RFn": 2, "Fin": 3}).astype(int)

all_data["Fence"] = all_data["Fence"].map(
        {'None': 0, "MnWw": 1, "GdWo": 2, "MnPrv": 3, "GdPrv": 4}).astype(int)
    

#%% 增加重要的特征
all_data['TotalSF_Bs+Gr'] = all_data['TotalBsmtSF'] + all_data['1stFlrSF'] + all_data['2ndFlrSF']
all_data['allArea'] = all_data['GrLivArea'] + all_data['GarageArea'] + all_data['TotalBsmtSF']
all_data['2ndFlrSF'] = pd.cut(all_data['2ndFlrSF'], bins=[-1,0,100000], labels=[0,1]).astype(int)
all_data['age'] = all_data['YrSold'] - all_data['YearRemodAdd']
all_data['newhouse'] = all_data['YrSold'] - all_data['YearBuilt']
all_data['newhouse'] = all_data['newhouse'].map(lambda x:1  if x < 3 else 0)
#all_data.drop(['YrSold', 'YearRemodAdd', 'YearBuilt'], axis=1, inplace=True)
all_data['Porch'] = all_data[['OpenPorchSF','EnclosedPorch','3SsnPorch','ScreenPorch']].sum(axis=1)

cols = ['ExterQual','BsmtQual','HeatingQC','KitchenQual','FireplaceQu','GarageQual','PoolQC']
Cond = ['BsmtCond', 'GarageCond','ExterCond']
all_data['Qu'] = all_data[cols].sum(axis=1)
all_data['Cond'] = all_data[Cond].sum(axis=1)

all_data["IsRegularLotShape"] = ~(all_data["LotShape"] == "Reg") * 1
all_data["IsLandLevel"] = ~(all_data["LandContour"] == "Lvl") * 1
all_data["IsLandSlopeGentle"] = ~(all_data["LandSlope"] == "Gtl") * 1
all_data["IsElectricalSBrkr"] = ~(all_data["Electrical"] == "SBrkr") * 1
all_data["IsGarageDetached"] = ~(all_data["GarageType"] == "Detchd") * 1
all_data["IsPavedDrive"] = ~(all_data["PavedDrive"] == "Y") * 1
all_data["HasShed"] = ~(all_data["MiscFeature"] == "Shed") * 1
all_data["Remodeled"] = ~(all_data["YearRemodAdd"] != all_data["YearBuilt"]) * 1
all_data["RecentRemodel"] = ~(all_data["YearRemodAdd"] == all_data["YrSold"]) * 1
all_data["VeryNewHouse"] = ~(all_data["YearBuilt"] == all_data["YrSold"]) * 1

all_data["HighSeason"] = all_data["MoSold"].replace(
        {1: 0, 2: 0, 3: 0, 4: 1, 5: 1, 6: 1, 7: 1, 8: 0, 9: 0, 10: 0, 11: 0, 12: 0})
all_data["NewerDwelling"] = all_data["MSSubClass"].replace(
        {20: 1, 30: 0, 40: 0, 45: 0,50: 0, 60: 1, 70: 0, 75: 0, 80: 0, 85: 0,
         90: 0, 120: 1, 150: 0, 160: 0, 180: 0, 190: 0}) 
    
all_data.loc[all_data.Neighborhood == 'NridgHt', "Neighborhood_Good"] = 1
all_data.loc[all_data.Neighborhood == 'Crawfor', "Neighborhood_Good"] = 1
all_data.loc[all_data.Neighborhood == 'StoneBr', "Neighborhood_Good"] = 1
all_data.loc[all_data.Neighborhood == 'Somerst', "Neighborhood_Good"] = 1
all_data.loc[all_data.Neighborhood == 'NoRidge', "Neighborhood_Good"] = 1
all_data["Neighborhood_Good"].fillna(0, inplace=True)

all_data["BoughtOffPlan"] = all_data.SaleCondition.replace(
        {"Abnorml" : 0, "Alloca" : 0, "AdjLand" : 0, "Family" : 0, "Normal" : 0, "Partial" : 1})

all_data["SeasonSold"] = all_data["MoSold"].map({12:0, 1:0, 2:0, 3:1, 4:1, 5:1, 
                                                  6:2, 7:2, 8:2, 9:3, 10:3, 11:3}).astype(int)
    
all_data["SimplOverallQual"] = all_data.OverallQual.replace(
        {1 : 1, 2 : 1, 3 : 1, 4 : 2, 5 : 2, 6 : 2, 7 : 3, 8 : 3, 9 : 3, 10 : 3})
all_data["SimplOverallCond"] = all_data.OverallCond.replace(
        {1 : 1, 2 : 1, 3 : 1, 4 : 2, 5 : 2, 6 : 2, 7 : 3, 8 : 3, 9 : 3, 10 : 3})
all_data["SimplPoolQC"] = all_data.PoolQC.replace(
        {1 : 1, 2 : 1, 3 : 2, 4 : 2})
all_data["SimplGarageCond"] = all_data.GarageCond.replace(
        {1 : 1, 2 : 1, 3 : 1, 4 : 2, 5 : 2})
all_data["SimplGarageQual"] = all_data.GarageQual.replace(
        {1 : 1, 2 : 1, 3 : 1, 4 : 2, 5 : 2})
all_data["SimplFireplaceQu"] = all_data.FireplaceQu.replace(
        {1 : 1, 2 : 1, 3 : 1, 4 : 2, 5 : 2})
all_data["SimplFireplaceQu"] = all_data.FireplaceQu.replace(
        {1 : 1, 2 : 1, 3 : 1, 4 : 2, 5 : 2})
all_data["SimplFunctional"] = all_data.Functional.replace(
        {1 : 1, 2 : 1, 3 : 2, 4 : 2, 5 : 3, 6 : 3, 7 : 3, 8 : 4})
all_data["SimplKitchenQual"] = all_data.KitchenQual.replace(
        {1 : 1, 2 : 1, 3 : 1, 4 : 2, 5 : 2})
all_data["SimplHeatingQC"] = all_data.HeatingQC.replace(
        {1 : 1, 2 : 1, 3 : 1, 4 : 2, 5 : 2})
all_data["SimplBsmtFinType1"] = all_data.BsmtFinType1.replace(
        {1 : 1, 2 : 1, 3 : 1, 4 : 2, 5 : 2, 6 : 2})
all_data["SimplBsmtFinType2"] = all_data.BsmtFinType2.replace(
        {1 : 1, 2 : 1, 3 : 1, 4 : 2, 5 : 2, 6 : 2})
all_data["SimplBsmtCond"] = all_data.BsmtCond.replace(
        {1 : 1, 2 : 1, 3 : 1, 4 : 2, 5 : 2})
all_data["SimplBsmtQual"] = all_data.BsmtQual.replace(
        {1 : 1, 2 : 1, 3 : 1, 4 : 2, 5 : 2})
all_data["SimplExterCond"] = all_data.ExterCond.replace(
        {1 : 1, 2 : 1, 3 : 1, 4 : 2, 5 : 2})
all_data["SimplExterQual"] = all_data.ExterQual.replace(
        {1 : 1, 2 : 1, 3 : 1, 4 : 2, 5 : 2})    
    
neighborhood_map = {
        "MeadowV" : 0,  #  88000
        "IDOTRR" : 1,   # 103000
        "BrDale" : 1,   # 106000
        "OldTown" : 1,  # 119000
        "Edwards" : 1,  # 119500
        "BrkSide" : 1,  # 124300
        "Sawyer" : 1,   # 135000
        "Blueste" : 1,  # 137500
        "SWISU" : 2,    # 139500
        "NAmes" : 2,    # 140000
        "NPkVill" : 2,  # 146000
        "Mitchel" : 2,  # 153500
        "SawyerW" : 2,  # 179900
        "Gilbert" : 2,  # 181000
        "NWAmes" : 2,   # 182900
        "Blmngtn" : 2,  # 191000
        "CollgCr" : 2,  # 197200
        "ClearCr" : 3,  # 200250
        "Crawfor" : 3,  # 200624
        "Veenker" : 3,  # 218000
        "Somerst" : 3,  # 225500
        "Timber" : 3,   # 228475
        "StoneBr" : 4,  # 278000
        "NoRidge" : 4,  # 290000
        "NridgHt" : 4,  # 315000
    }

all_data["NeighborhoodBin"] = all_data["Neighborhood"].map(neighborhood_map)





#%% 删除的变量
all_data.drop('BsmtUnfSF', axis=1, inplace=True) #删除后有提升
#all_data.drop('OverallCond', axis=1, inplace=True) #删除后有提升
all_data.drop('OpenPorchSF', axis=1, inplace=True) #删除后有提升
all_data.drop('EnclosedPorch', axis=1, inplace=True) #删除后有提升
all_data.drop('3SsnPorch', axis=1, inplace=True) #删除后有提升
all_data.drop('ScreenPorch', axis=1, inplace=True) #删除后有提升

all_data.drop('MoSold', axis=1, inplace=True) #删除后有提升
all_data.drop('YrSold', axis=1, inplace=True) #删除后有提升

#%% 离散数值变量，分箱
#对YearBuilt
bins = range(all_data['YearBuilt'].min()-1, all_data['YearBuilt'].max()+50, 3)
labels = range(len(bins)-1)
all_data['YearBuilt'] = pd.cut(all_data['YearBuilt'], bins, labels=labels) #cut是cate属性
all_data['YearBuilt'] = all_data['YearBuilt'].astype(int)
#YearRemodAdd
bins = range(all_data['YearRemodAdd'].min()-1, all_data['YearRemodAdd'].max()+50, 3)
labels = range(len(bins)-1)
all_data['YearRemodAdd'] = pd.cut(all_data['YearRemodAdd'], bins, labels=labels) #cut是cate属性
all_data['YearRemodAdd'] = all_data['YearRemodAdd'].astype(int)


#%% 独热编码
cols = ['MSSubClass', 'MSZoning', '']
all_data = pd.get_dummies(all_data)

#%% 
#all_data.plot(kind='scatter', x='WoodDeckSF', y='SalePrice')
#train_data.groupby('MoSold')['SalePrice'].mean()

#%% 数据标准化
#from sklearn.preprocessing import StandardScaler
#scaler = StandardScaler()
##
#cols = ['index', 'Id']
#numeric_features = list(all_data.dtypes[all_data.dtypes != "object"].index)
#for each in cols:
#    numeric_features.remove(each)
#scaler.fit(all_data[numeric_features])
#scaled = scaler.transform(all_data[numeric_features])
#for i, col in enumerate(numeric_features):
#    all_data[col] = scaled[:, i]
##



#%% 
X_train = all_data[all_data['SalePrice'].notnull()].reset_index(drop=True)
X_test = all_data[all_data['SalePrice'].isnull()].drop(['SalePrice'], axis=1).reset_index(drop=True)
     
X_train.drop(['index', 'Id'], axis=1, inplace=True)
X_test_Id = X_test['Id']
X_test.drop(['index', 'Id'], axis=1, inplace=True)


Y_train = X_train.pop('SalePrice')




#%% 使用xgboost预测
from sklearn.model_selection import GroupKFold
from sklearn.preprocessing import minmax_scale
import lightgbm as lgb


#train_data = lgb.Dataset(X_train, 
#                         label=Y_train,
#                         feature_name=list(X_train.columns), 
#                         categorical_feature=[cate_cols])


#几个重要参数
#针对 Leaf-wise (最佳优先) 树的参数优化
#num_leaves 一棵树上的叶子数
#min_data_in_leaf 一个叶子上数据的最小数量. 可以用来处理过拟合.默认20
#max_depth 限制树模型的最大深度


#针对更快的训练速度
#通过设置 bagging_fraction 和 bagging_freq 参数来使用 bagging 方法
#通过设置 feature_fraction 参数来使用特征的子抽样
#使用较小的 max_bin
#使用 save_binary 在未来的学习过程对数据加载进行加速


#%%针对更好的准确率
#使用较大的 max_bin （学习速度可能变慢）
#使用较小的 learning_rate 和较大的 num_iterations
#使用较大的 num_leaves （可能导致过拟合）
#尝试 dart


#%% 处理过拟合
#使用较小的 max_bin
#使用较小的 num_leaves
#使用 min_data_in_leaf 和 min_sum_hessian_in_leaf
#通过设置 bagging_fraction 和 bagging_freq 来使用 bagging
#通过设置 feature_fraction 来使用特征子抽样
#使用更大的训练数据
#使用 lambda_l1, lambda_l2 和 min_gain_to_split 来使用正则
#尝试 max_depth 来避免生成过深的树
params={'learning_rate': 0.01,    #学习速率
        'n_estimators': 1910, 
        'num_leaves': 30, 
        'max_depth': 4,          #叶的最大深度
        'min_data_in_leaf': 8,  
        'metric':'rmse',          #目标函数
        'objective': 'regression', 
        'verbose': 0,
        'bagging_fraction': 0.5,    #每次迭代用的数据比例
        'feature_fraction': 0.6,     #每次迭代用的特征比例
        'reg_alpha': 1,
        'reg_lambda': 0.01,
        'random_state': 8}


#data_train = lgb.Dataset(X_train, Y_train, silent=True)
#cv_results = lgb.cv(
#    params, data_train, num_boost_round=10000, nfold=5, stratified=False, shuffle=True, metrics='rmse',
#    early_stopping_rounds=50, verbose_eval=100, show_stdv=True)
#
#parameters = {
#              'reg_alpha':[0.001,0.01,0.1,1],
#              'reg_lambda':[0.001,0.01,0.1,1],
#              }
#
reg = lgb.LGBMRegressor(**params)
#gsearch = GridSearchCV(reg, param_grid=parameters, scoring='neg_mean_squared_error', cv=5)
#gsearch.fit(X_train, Y_train)
#gsearch.best_params_


reg.fit(X_train, Y_train)
result_xgb = reg.predict(X_test)
result_xgb = pd.DataFrame(result_xgb)
result_xgb = pd.concat([X_test_Id, result_xgb], axis=1)
result_xgb.columns = ['Id', 'SalePrice']


a = pd.DataFrame(reg.feature_importances_, index=X_train.columns).sort_values(by=0, ascending=False)
print('xgboost得分为:', rmsle_cv(reg, X_train, Y_train))
result_xgb.to_csv('submission_xgb.csv', index=False)


#%% 使用Lasso
#from sklearn.linear_model import Lasso
#best_alpht = 0.00099
#lasso = Lasso(alpha=best_alpht, max_iter=50000)
#lasso.fit(X_train, Y_train)
#
#result_lasoo = lasso.predict(X_test)
#print('lasso得分为:', rmsle_cv(lasso, X_train, Y_train))



















