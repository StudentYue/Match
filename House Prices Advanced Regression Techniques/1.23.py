# -*- coding: utf-8 -*-

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import pandas as pd 
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from sklearn.metrics import mean_squared_error
import math
#%% 缺失值处理
train_data = pd.read_csv(r"C:\Users\ZHOU-JC\Desktop\data match\House Prices Advanced Regression Techniques/train.csv")
test_data  = pd.read_csv(r"C:\Users\ZHOU-JC\Desktop\data match\House Prices Advanced Regression Techniques/test.csv")
all_data = train_data.append(test_data, sort=False).reset_index()
#%%

total = all_data.isnull().sum().sort_values(ascending=False)
percent = (all_data.isnull().sum())/all_data.isnull().count().sort_values(ascending=False)
missing_data = pd.concat([total, percent], axis=1, sort=False, keys=['Total', 'Percent'])
missing_data[missing_data['Percent'] > 0]

# PoolQc 泳池面积
all_data.drop('PoolQC', axis=1, inplace=True)
# MiscFeature 其他特征
all_data.drop('MiscFeature', axis=1, inplace=True)
# Alley 巷子类别
all_data.drop('Alley', axis=1, inplace=True)
# Fence 围墙质量
all_data.drop('Fence', axis=1, inplace=True)
# FireplaceQu 壁炉质量 Fireplaces壁炉数量为0的话 也就没有壁炉了
all_data.loc[all_data['Fireplaces']==0, 'FireplaceQu'] = 'none'
# LotFrontage 距离街道的直线距离 
all_data.drop('LotFrontage', axis=1, inplace=True)
# GarageQual 车库质量
all_data['GarageQual'].fillna('none', inplace=True)
# GarageYrBlt 车库建造年份
all_data.drop('GarageYrBlt', axis=1, inplace=True)
# GarageFinish 车库内饰
all_data['GarageFinish'].fillna('none', inplace=True)
# GarageCond 车库条件
all_data['GarageCond'].fillna('none', inplace=True)
# GarageType 车库类型 一般 删除吧
all_data.drop('GarageType', axis=1, inplace=True)
# BsmtExposure 花园地下室墙 删除
all_data.drop('BsmtExposure', axis=1 , inplace=True)
# BsmtCond 地下室概况
all_data['BsmtCond'].fillna('none', inplace=True)
# BsmtQual 地下室高度
all_data['BsmtQual'].fillna('none', inplace=True)
# BsmtFinType2 地下室装饰质量
# BsmtFinType1 地下室装饰质量
all_data.drop(['BsmtFinType2','BsmtFinType1'], axis=1, inplace=True)
# MasVnrType 砌体饰面类型
all_data.drop('MasVnrType', axis=1, inplace=True)
# MasVnrArea 砌体饰面面积
all_data.drop('MasVnrArea', axis=1, inplace=True)
# MSZoning 区域分类
all_data.drop('MSZoning', axis=1, inplace=True)
# Utilities 公共设施类型
all_data.drop('Utilities', axis=1, inplace=True)
# Functional 房屋功能性评级
all_data.drop('Functional', axis=1, inplace=True)
# BsmtHalfBath 地下室半浴室
all_data['BsmtHalfBath'].fillna(0, inplace=True)
# BsmtFullBath 地下室全浴室
all_data['BsmtFullBath'].fillna(0, inplace=True)
# GarageArea 车库面积
all_data['GarageArea'].fillna(0, inplace=True)
# GarageCars 车库车容量大小
all_data['GarageCars'].fillna(0, inplace=True)
# Exterior2nd 住宅外墙
# Exterior1st 住宅外墙
all_data.drop('Exterior2nd', axis=1, inplace=True)
all_data.drop('Exterior1st', axis=1, inplace=True)
# SaleType 交易类型
all_data['SaleType']=all_data['SaleType'].map(lambda x:'WD' if x=='WD' else 'other')
# BsmtFinSF1 地下室装饰面积
all_data['BsmtFinSF1'].fillna(0, inplace=True)
# BsmtFinSF2 地下室装饰面积 
all_data['BsmtFinSF2'].fillna(0, inplace=True)
# TotalBsmtSF 地下室总面积
all_data['TotalBsmtSF'].fillna(0, inplace=True)
# BsmtUnfSF 地下室未装饰面积
all_data.drop('BsmtUnfSF', axis=1, inplace=True)
# Electrical 电力系统
all_data['Electrical']=all_data['Electrical'].map(lambda x:'SBrkr' if x=='SBrkr' else 'other')
# KitchenQual 厨房质量
all_data['KitchenQual'].fillna('TA', inplace=True)

#%% 查看SalePrice 特殊值
all_data = all_data.drop(all_data[all_data['Id'] == 1299].index)
all_data = all_data.drop(all_data[all_data['Id'] == 524 ].index)

#%% 删除不重要的变量
cols = ['LotFrontage', 'Street', 'Alley', 'Utilities', 'LandSlope', 'RoofMatl',
        'ExterCond', 'Heating', 'LowQualFinSF', 'PoolArea', 
        'PoolQc', 'Fence', 'MiscFeature', 'MiscVal']
for each in cols:
    if each in all_data.columns:
        all_data.drop(each, axis=1, inplace=True)
        
#%% 查看特征中类别太多的变量
cate_cols = all_data.select_dtypes(include='object').columns.tolist()

#%% LotShape
all_data['LotShape'] = all_data['LotShape'].map(lambda x:'Reg' 
                                                if x == 'Reg' else 'other')
# LandContour
all_data.drop('LandContour', axis=1, inplace=True)
# Neighborhood
all_data.drop('Neighborhood', axis=1, inplace=True)
# Condition1
all_data.drop('Condition1', axis=1, inplace=True)
# Condition2
all_data.drop('Condition2', axis=1, inplace=True)
# BldgType
all_data.drop('BldgType', axis=1, inplace=True)
# HouseStyle
all_data.drop('HouseStyle', axis=1, inplace=True)
# RoofStyle
all_data.drop('RoofStyle', axis=1, inplace=True)

# ExterQual
all_data['ExterQual'] = all_data['ExterQual'].map(lambda x:4 if x == 'Ex' 
                                                  else 3 if x == 'Gd'
                                                 else 2 if x =='TA'
                                                 else 1)
# Foundation
all_data['Foundation'] = all_data['Foundation'].map(lambda x:'PConc' 
                                                if x == 'PConc' else 'other')
# BsmtQual
all_data['BsmtQual'] = all_data['BsmtQual'].map(lambda x:4 if x == 'Ex' 
                                                  else 3 if x == 'Gd'
                                                 else 2 if x =='TA'
                                                 else 1)
# BsmtCond
all_data['BsmtCond'] = all_data['BsmtCond'].map(lambda x:4 if x == 'Gd' 
                                                  else 3 if x == 'TA'
                                                 else 2 if x =='Fa'
                                                 else 1)
# BsmtExposure
# BsmtFinType1
#all_data['BsmtFinType1'] = all_data['BsmtFinType1'].map(lambda x:'GLQ' if x == 'GLQ' 
#                                                  else 'other')

# HeatingQC
all_data['HeatingQC'] = all_data['HeatingQC'].map(lambda x:4 if x == 'Ex' 
                                                  else 3 if x == 'Gd'
                                                 else 2 if x =='TA'
                                                 else 1 if x =='Fa'
                                                 else 0)
# CentralAir
# Electrical
all_data['Electrical'] = all_data['Electrical'].map(lambda x:'SBrkr' if x == 'SBrkr' 
                                                  else 'other')
# KitchenQual
all_data['KitchenQual'] = all_data['KitchenQual'].map(lambda x:4 if x == 'Ex' 
                                                  else 3 if x == 'Gd'
                                                 else 2 if x =='TA'
                                                 else 1)
# FireplaceQu
all_data['FireplaceQu'] = all_data['FireplaceQu'].map(lambda x:4 if x == 'Ex' 
                                                  else 3 if x == 'Gd'
                                                 else 2 if x =='TA'
                                                 else 1 if x =='Fa'
                                                 else 0)


# GarageFinish
# GarageQual
all_data['GarageQual'] = all_data['GarageQual'].map(lambda x:4 if x == 'Ex' 
                                                  else 3 if x == 'Gd'
                                                 else 2 if x =='TA'
                                                 else 1 if x =='Fa'
                                                 else 0)
# GarageCond
all_data['GarageCond'] = all_data['GarageCond'].map(lambda x:4 if x == 'Ex' 
                                                  else 3 if x == 'Gd'
                                                 else 2 if x =='TA'
                                                 else 1 if x =='Fa'
                                                 else 0)
# PavedDrive
all_data['PavedDrive'] = all_data['PavedDrive'].map(lambda x:'Y' if x == 'Y' 
                                                  else 'other')

# SaleType
all_data.drop('SaleType', axis=1, inplace=True)
# SaleCondition
all_data['SaleCondition'] = all_data['SaleCondition'].map(lambda x:'Partial' if x == 'Partial' 
                                                  else 'other')

        

#%% 变成0——1变量的 (还未独热编码)
col_0_1= ['WoodDeckSF']

for each in col_0_1:
    all_data[each] = all_data[each].map(lambda x:0 if x==0 else 1)
    
all_data['Porch'] = all_data['OpenPorchSF'] + all_data['EnclosedPorch'] + all_data['3SsnPorch'] + all_data['ScreenPorch']
# =============================================================================
#%% 总面积
# GrLivArea居住面积 加 GaraAre 车库面积 加 TotalBsmtSF地下室面积
all_data['allArea'] = all_data['GrLivArea'] + all_data['GarageArea'] + all_data['TotalBsmtSF']
# 将二层面积 2stFlrSF 转为0-1变量        
all_data['2ndFlrSF'] = pd.cut(all_data['2ndFlrSF'], bins=[-1,0,100000], labels=[0,1])
# 删除一层面积
all_data['1st_per'] = all_data['1stFlrSF'] / all_data['GrLivArea']
# 地下室已装饰面积占地下室总面积的比例  部分地下室面积为0 不能这样除
#all_data['SFed'] = all_data['BsmtFinSF1'] / all_data['TotalBsmtSF'] 
# 门廊 
#all_data['Porch'] = all_data['OpenPorchSF'].values +all_data['EnclosedPorch'].values+all_data['3SsnPorch'].values+all_data['ScreenPorch'].values
# 所有质量评分
cols = ['ExterQual','BsmtQual','HeatingQC','KitchenQual','FireplaceQu','GarageQual']
Cond = ['BsmtCond', 'GarageCond']
all_data['QU'] = all_data[cols].sum(axis=1)
all_data['Cond'] = all_data[Cond].sum(axis=1)

#%% 删除LotArea
all_data.drop('LotArea', axis=1, inplace=True)
# 对重要几个特征进行排名
cols = ['allArea', 'GrLivArea', 'GarageArea', '1stFlrSF',  'TotalBsmtSF']
#for each in cols:
#    all_data['rank.'+each] = all_data[each].rank(pct=True)
#    all_data.drop(each, axis=1, inplace=True)
    
# 总的浴室数量
all_data['allTolBathrooms'] = all_data['FullBath'] + all_data['HalfBath'] + all_data['BsmtFullBath'] + all_data['BsmtHalfBath']
cols = ['FullBath', 'HalfBath', 'BsmtFullBath', 'BsmtHalfBath']
all_data.drop(cols, axis=1, inplace=True)
    
# 对房屋年龄有关的几个向量
# 房屋年龄
all_data['age'] = all_data['YrSold'] - all_data['YearRemodAdd']
# 房屋有没有翻新过
all_data['novation'] = all_data['YearRemodAdd'] - all_data['YearBuilt']
all_data['novation'] = all_data['novation'].map(lambda x:0 if x==0 else 1)
# 是否是新房子 直接卖的那种
all_data['newhouse'] = all_data['YrSold'] - all_data['YearBuilt']
all_data['newhouse'] = all_data['newhouse'].map(lambda x:1  if x < 3 else 0)

all_data.drop(['YrSold', 'YearRemodAdd', 'YearBuilt'], axis=1, inplace=True)

#all_data.drop(cols, axis=1, inplace=True)
# =============================================================================


#%% 独热编码
cate_cols = list(all_data.select_dtypes(include='object').columns)
for each in cate_cols:
    all_data = pd.concat([all_data, pd.get_dummies(all_data[each], prefix=each)], axis=1)
    all_data.drop(each, axis=1, inplace=True)      
    
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

params={'learning_rate': 0.1,    #学习速率
        'n_estimators': 500, 
        'max_depth': None,          #叶的最大深度
        'metric':'rmse',          #目标函数
        'num_leaves': 31,
        'verbose': 0,
        'bagging_fraction': 0.8,    #每次迭代用的数据比例
        'feature_fraction': 0.8,     #每次迭代用的特征比例
        'random_state': 10}
    
reg = lgb.LGBMRegressor(**params)
reg.fit(X_train, Y_train)
result_xgb = reg.predict(X_test)
result_xgb = pd.DataFrame(result_xgb)
result_xgb = pd.concat([X_test_Id, result_xgb], axis=1)
result_xgb.columns = ['Id', 'SalePrice']

a = pd.DataFrame(reg.feature_importances_, index=X_train.columns).sort_values(by=0, ascending=False)
print('xgboost得分为:',math.sqrt(mean_squared_error(reg.predict(X_train), Y_train))/Y_train.mean())
result_xgb.to_csv('submission_xgb.csv', index=False)

#%% 使用随机森林预测
from sklearn.ensemble import RandomForestRegressor

params={'n_estimators': 500,
        'max_features': 0.8, #选择最适属性时划分的特征不能超过此值。
        'max_depth': 20,          #叶的最大深度   
        'min_samples_split': 2, #似乎没影响， 根据属性划分节点时，每个划分最少的样本数 
        'min_samples_leaf': 1, #似乎没影响， 叶子节点最少的样本数
#        'max_leaf_nodes': (default=None)叶子树的最大样本数。
#        'min_weight_fraction_leaf': (default=0) 叶子节点所需要的最小权值
        'n_jobs':-1,
        'verbose': 0}

rf = RandomForestRegressor(**params, random_state=10)
rf.fit(X_train, Y_train)
result_rf = rf.predict(X_test)
result_rf = pd.DataFrame(result_rf)
result_rf = pd.concat([X_test_Id, result_rf], axis=1)
result_rf.columns = ['Id', 'SalePrice']
print('随机森林得分为:',math.sqrt(mean_squared_error(rf.predict(X_train), Y_train))/Y_train.mean())
result_rf.to_csv('submission_rf.csv', index=False)

#%% gdbt
params={'learning_rate': 0.1,
        'n_estimators': 500,
        'subsample': 0.8,
        'loss': 'ls', 
        #对于回归模型，有均方差"ls",
        #绝对损失"lad", Huber损失"huber"和分位数损失“quantile”。
        #默认是均方差"ls"。一般来说，如果数据的噪音点不多，用默认的均方差"ls"比较好。如果是噪音点较多，则推荐用抗噪音的损失函数"huber"。而如果我们需要对训练集进行分段预测的时候，则采用“quantile”。
        'max_features': 0.7, #选择最适属性时划分的特征不能超过此值。
        'max_depth': None,          #叶的最大深度   
        'min_samples_split': 2, #似乎没影响， 根据属性划分节点时，每个划分最少的样本数 
        'min_samples_leaf': 1, #似乎没影响， 叶子节点最少的样本数
#        'max_leaf_nodes': (default=None)叶子树的最大样本数。
#        'min_weight_fraction_leaf': (default=0) 叶子节点所需要的最小权值
        'verbose': 0}
from sklearn.ensemble import GradientBoostingRegressor
gbdt = GradientBoostingRegressor(**params, random_state=10)
gbdt.fit(X_train, Y_train)
result_gbdt = rf.predict(X_test)
result_gbdt = pd.DataFrame(result_gbdt)
result_gbdt = pd.concat([X_test_Id, result_gbdt], axis=1)
result_gbdt.columns = ['Id', 'SalePrice']
print('gdbt:',math.sqrt(mean_squared_error(gbdt.predict(X_train), Y_train))/Y_train.mean())




#%% 输出
result = pd.concat([result_xgb['Id'], (result_xgb['SalePrice']+result_rf['SalePrice']+result_gbdt['SalePrice'])/3], axis=1)
#result.to_csv('submission.csv', index=False)

print('融合后:',math.sqrt(mean_squared_error((reg.predict(X_train)+
                                              rf.predict(X_train)+
                                              gbdt.predict(X_train))/3, Y_train))/Y_train.mean())
result_gbdt.to_csv('submission_gbdt.csv', index=False)





#%%

a = pd.DataFrame(reg.feature_importances_, index=X_train.columns).sort_values(by=0, ascending=False)

#%% 
cols = ['ExterQual','BsmtQual','BsmtCond','HeatingQC','KitchenQual','FireplaceQu','GarageQual','GarageCond']
b = all_data[cols].sum(axis=1)

#%%
from sklearn.model_selection import KFold
kf = KFold(n_splits=5, shuffle=True)
for i, (train_index,test_index) in enumerate(kf.split(X_train)):
    x = train_index
    y = test_index





# -*- coding: utf-8 -*-
"""
Created on Thu Jan 24 09:20:36 2019

@author: ZHOU-JC
"""

#%% stacking
#第一层学习器
#
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.ensemble import RandomForestRegressor
import lightgbm as lgb
from sklearn.model_selection import KFold
#%% 切分
kf = KFold(n_splits=5, shuffle=True)

clfs = [
        GradientBoostingRegressor(),
        RandomForestRegressor(),
        lgb.LGBMRegressor()]


# 对于每一层回归，skf为fold数
blend_train  = np.zeros((X_train.shape[0], len(clfs)))
blend_test   = np.zeros((X_test.shape[0],   len(clfs)))
for j, clf in enumerate(clfs):
    print('正在用第一层学习器的第%s个'%j)
    blend_test_j = np.zeros((X_test.shape[0], 5))
    
    # Number of testing data x Number of folds , we will take the mean of the predictions later
    for i, (train_index, cv_index) in enumerate(kf.split(X_train)):
#        print('Fold [%s]' % (i))
            
        # This is the training and validation set
        x_train = X_train.iloc[train_index]
        y_train = Y_train.iloc[train_index]
        x_cv    = X_train.iloc[cv_index]
        y_cv    = Y_train.iloc[cv_index]
        
        clf.fit(x_train, y_train)
            
        # This output will be the basis for our blended classifier to train against,
        # which is also the output of our classifiers
        blend_train[cv_index, j] = clf.predict(x_cv)
        blend_test_j[:, i] = clf.predict(X_test)        
    # Take the mean of the predictions of the cross validation set
    blend_test[:, j] = blend_test_j.mean(1)
   

#%% 3. 接着用 blend_train, Y_dev 去训练第二层的学习器 LogisticRegression：
# 接着用 blend_train, Y_dev 去训练第二层的学习器 LogisticRegression：
# Start blending!
lasso.fit(blend_train, Y_train)

#%% 4. 再用 bclf 来预测测试集 blend_test，并得到 score：
# Predict now
Y_test_predict = lasso.predict(blend_test)

result_stacking = pd.concat([test_data['Id'], pd.DataFrame(Y_test_predict)], axis=1)

result_stacking.columns = ['Id', 'SalePrice']
result_stacking.to_csv('submission_stacking.csv', index=False)