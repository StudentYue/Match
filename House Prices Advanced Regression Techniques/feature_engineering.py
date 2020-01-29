# -*- coding: utf-8 -*-
"""
Created on Thu Nov 22 19:55:48 2018

@author: ZHOU-JC
"""

import pandas as pd
import numpy as np
import seaborn as sns
from scipy import stats
from scipy.stats import skew
from scipy.stats import norm
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.manifold import TSNE
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
import seaborn as sns
train_data = pd.read_csv(r"C:\Users\ZHOU-JC\Desktop\data match\House Prices Advanced Regression Techniques/train.csv")
test_data = pd.read_csv(r"C:\Users\ZHOU-JC\Desktop\data match\House Prices Advanced Regression Techniques/test.csv")

#%%
all_data = train_df.append(test_df).reset_index()
numcols = list(all_data.select_dtypes(include='number').columns.values)
catebols = [each for each in list(all_data.columns) if each not in numcols]

#%%
a = all_data.describe().T
#%% MSSubClass 建筑等级 
train_data['MSSubClass'].plot(kind='hist', bins=50)
train_data.plot(kind='scatter', x='MSSubClass', y='SalePrice')


#%% MSZoing 社区分类 重要
train_data['MSZoning'].value_counts()
train_data.groupby('MSZoning')['SalePrice'].mean().plot(kind='bar')


#%% LotFrontage 离贫民窟的距离 有缺失值
train_data['LotFrontage'].plot(kind='hist', bins=50)
#train_data.plot(kind='scatter', x='LotFrontage', y='SalePrice')
train_data['LotFrontage_0'] = pd.cut(train_data['LotFrontage'], bins=50)
train_data.groupby('LotFrontage_0')['SalePrice'].mean().plot(kind='bar')

#%% LotArea 面积
train_data['LotArea'].plot(kind='hist', bins=50)
train_data.plot(kind='scatter', x='LotArea', y='SalePrice')

#%% Street:  Type of road access
# 不平衡变量 超过99%是同一个值
#%% Aeely: Type of alley access 
# 缺失值好多
#%% LotShape： 房子整体形状，类型：类别型
train_data['LotShape'].value_counts()
train_data.groupby('LotShape')['SalePrice'].mean().plot(kind='bar')

#%% LandContour: 平整度级别，类型：类别型
train_data['LandContour'].value_counts()
train_data.groupby('LandContour')['SalePrice'].mean().plot(kind='bar')

#%% Utilities: Type of utilities available
train_data['Utilities'].value_counts()
train_data.groupby('Utilities')['SalePrice'].mean().plot(kind='bar')
# 不平衡变量
#%% LandSlope: Slope of property
train_data['LandSlope'].value_counts()
train_data.groupby('LandSlope')['SalePrice'].mean().plot(kind='bar')


#%% LotShape 影响不大
train_data['LotShape'].value_counts()
train_data.groupby('LotShape')['SalePrice'].mean().plot(kind='bar')

#%% LandCountour: Flatness of the property
# 有点用 但不是差别很大
train_data['LandContour'].value_counts()
train_data.groupby('LandContour')['SalePrice'].mean().plot(kind='bar')

#%% LotConfig： Lot configuration
train_data['LotConfig'].value_counts()
train_data.groupby('LotConfig')['SalePrice'].mean().plot(kind='bar')

#%% LandSlope: Slope of property
train_data['LandSlope'].value_counts()
train_data.groupby('LandSlope')['SalePrice'].mean().plot(kind='bar')

#%% Neighborhood: Physical locations within Ames city limits
train_data['Neighborhood'].value_counts()
train_data.groupby('Neighborhood')['SalePrice'].mean().plot(kind='bar')

#%%Condition1: Proximity to main road or railroad
# Condition2: Proximity to main road or railroad (if a second is present)
# 感觉没什么用
# 离主干道和铁路的距离
train_data['Condition2'].value_counts()
train_data.groupby('Condition2')['SalePrice'].mean().plot(kind='bar')

#%% BldgType: Type of dwelling
train_data['BldgType'].value_counts()
train_data.groupby('BldgType')['SalePrice'].mean().plot(kind='bar')

#%% HouseStyle: Style of dwelling 有用
train_data['HouseStyle'].value_counts()
train_data.groupby('HouseStyle')['SalePrice'].mean().plot(kind='bar')

# =============================================================================
# 很重要的变量
#%% OverallQual: Overall material and finish quality
#%% OverallCond: Overall condition rating
train_data['OverallQual'].value_counts()
train_data.groupby('OverallQual')['SalePrice'].mean().plot(kind='bar')
train_data['OverallCond'].value_counts()
train_data.groupby('OverallCond')['SalePrice'].mean().plot(kind='bar')
#%% YearBuilt: Original construction date
#%% YearRemodAdd: Remodel date
train_data['YearBuilt'].plot(kind='hist', bins=50)
train_data.plot(kind='scatter', x='YearBuilt', y='SalePrice')
train_data['YearRemodAdd'].plot(kind='hist', bins=50)
train_data.plot(kind='scatter', x='YearRemodAdd', y='SalePrice')

#%% Exterior1st: Exterior covering on house
#%% Exterior2nd: Exterior covering on house (if more than one material)
train_data['Exterior1st'].value_counts()
train_data.groupby('Exterior1st')['SalePrice'].mean().plot(kind='bar')
train_data['Exterior2nd'].value_counts()
train_data.groupby('Exterior2nd')['SalePrice'].mean().plot(kind='bar')

#%%ExterQual: Exterior material quality
#%%ExterCond: Present condition of the material on the exterior
train_data['ExterQual'].value_counts()
train_data.groupby('ExterQual')['SalePrice'].mean().plot(kind='bar')
train_data['ExterCond'].value_counts()
train_data.groupby('ExterCond')['SalePrice'].mean().plot(kind='bar')

#%% Foundation: Type of foundation
train_data['Foundation'].value_counts()
train_data.groupby('Foundation')['SalePrice'].mean().plot(kind='bar')
# =============================================================================


#%% RoofStyle: Type of roof
#%% RoofMatl: Roof material #非平衡特征
train_data['RoofStyle'].value_counts()
train_data.groupby('RoofStyle')['SalePrice'].mean().plot(kind='bar')
train_data['RoofMatl'].value_counts()
train_data.groupby('RoofMatl')['SalePrice'].mean().plot(kind='bar')

#%% Exterior1st: Exterior covering on house
#%% Exterior2nd: Exterior covering on house (if more than one material)
train_data['Exterior1st'].value_counts()
train_data.groupby('Exterior1st')['SalePrice'].mean().plot(kind='bar')
train_data['Exterior2nd'].value_counts()
train_data.groupby('Exterior2nd')['SalePrice'].mean().plot(kind='bar')

#%%MasVnrType: Masonry veneer type
#%%MasVnrArea: Masonry veneer area in square feet
train_data['MasVnrType'].value_counts()
train_data.groupby('MasVnrType')['SalePrice'].mean().plot(kind='bar')
train_data['MasVnrArea_0'] = pd.cut(train_data['MasVnrArea'], bins=[-1,0,10000])
train_data['MasVnrArea'].plot(kind='hist',bins=50)
train_data.plot(kind='scatter', x='MasVnrArea', y='SalePrice')

#%%BsmtQual: Height of the basement #重要
#%%BsmtCond: General condition of the basement #一般
#%%BsmtExposure: Walkout or garden level basement walls #一般
#%%BsmtFinType1: Quality of basement finished area #重要
#%%BsmtFinSF1: Type 1 finished square feet #重要
#%%BsmtFinType2: Quality of second finished area (if present) #一般
#%%BsmtFinSF2: Type 2 finished square feet #一般
#%%BsmtUnfSF: Unfinished square feet of basement area #一般
#%%TotalBsmtSF: Total square feet of basement area #重要
train_data['BsmtFinType2'].value_counts()
train_data.groupby('BsmtFinType2')['SalePrice'].mean().plot(kind='bar')
train_data['TotalBsmtSF'].plot(kind='hist', bins=50)
train_data.plot(kind='scatter', x='TotalBsmtSF', y='SalePrice')

#%% Heating: Type of heating #不平衡
#%% HeatingQC: Heating quality and condition #很重要
train_data['HeatingQC'].value_counts()
train_data.groupby('HeatingQC')['SalePrice'].mean().plot(kind='bar')

#%% CentralAir: Central air conditioning #不平衡，但有用
train_data['CentralAir'].value_counts()
train_data.groupby('CentralAir')['SalePrice'].mean().plot(kind='bar')

#%% Electrical: Electrical system #不平衡，但有用
train_data['Electrical'].value_counts()
train_data.groupby('Electrical')['SalePrice'].mean().plot(kind='bar')

#%% 1stFlrSF: First Floor square feet #重要
#%% 2ndFlrSF: Second floor square feet #一般 可以分成有没有二层
train_data['2ndFlrSF'].plot(kind='hist', bins=50)
train_data['2ndFlrSF_0'] = pd.cut(train_data['2ndFlrSF'], bins=[-1,0,2000])
train_data.plot(kind='scatter', x='TotalBsmtSF', y='SalePrice')
#%% LowQualFinSF: Low quality finished square feet (all floors) #不平衡
train_data['LowQualFinSF'].value_counts()


#%% GrLivArea: Above grade (ground) living area square feet #很重要
train_data['GrLivArea'].plot(kind='hist', bins=50) 
train_data.plot(kind='scatter', x='GrLivArea', y='SalePrice')

#%% BsmtFullBath: Basement full bathrooms #重要
#%% BsmtHalfBath: Basement half bathrooms #一般 
#%% FullBath: Full bathrooms above grade #重要
train_data['FullBath'].value_counts()
train_data.groupby('FullBath')['SalePrice'].mean().plot(kind='bar')

#%% HalfBath: Half baths above grade #重要
#%% BedroomAbvGr: Number of bedrooms above basement level #一般
train_data['BedroomAbvGr'].plot(kind='hist', bins=50) 
train_data.groupby('BedroomAbvGr')['SalePrice'].mean().plot(kind='bar')

#%% KitchenAbvGr: Number of kitchens #不太平衡，但很有用
train_data['KitchenAbvGr'].plot(kind='hist', bins=50) 
train_data.groupby('KitchenAbvGr')['SalePrice'].mean().plot(kind='bar')

#%% KitchenQual: Kitchen quality 重要
train_data['KitchenQual'].value_counts()
train_data.groupby('KitchenQual')['SalePrice'].mean().plot(kind='bar')

#%% TotRmsAbvGrd: Total rooms above grade (does not include bathrooms) #很重要
train_data['TotRmsAbvGrd'].value_counts()
train_data.groupby('TotRmsAbvGrd')['SalePrice'].mean().plot(kind='bar')
 
#%% Functional: Home functionality rating #不太平衡，但蛮有用
train_data['Functional'].value_counts()
train_data.groupby('Functional')['SalePrice'].mean().plot(kind='bar')

#%% Fireplaces: Number of fireplaces #重要
#%% FireplaceQu: Fireplace quality #重要
train_data['Fireplaces'].value_counts()
train_data.groupby('Fireplaces')['SalePrice'].mean().plot(kind='bar')
train_data['FireplaceQu'].value_counts()
train_data.groupby('FireplaceQu')['SalePrice'].mean().plot(kind='bar')

#%% GarageType: Garage location #一般
#%% GarageYrBlt: Year garage was built #重要
#%% GarageFinish: Interior finish of the garage #重要
#%% GarageCars: Size of garage in car capacity #重要
#%% GarageArea: Size of garage in square feet #重要
#%% GarageQual: Garage quality #不平衡 但重要
#%% GarageCond: Garage condition #不平衡，但重要
train_data['GarageCond'].value_counts()
train_data.groupby('GarageCond')['SalePrice'].mean().plot(kind='bar')
train_data['GarageArea'].plot(kind='hist', bins=50)
train_data.plot(kind='scatter', x='GarageArea', y='SalePrice')

#%% PavedDrive: Paved driveway #重要
train_data['PavedDrive'].value_counts()
train_data.groupby('PavedDrive')['SalePrice'].mean().plot(kind='bar')

#%% WoodDeckSF: Wood deck area in square feet #可以分成0-1变量
#%% OpenPorchSF: Open porch area in square feet #可以分成0-1变量
#%% EnclosedPorch: Enclosed porch area in square feet #可以分成0-1变量
#%% 3SsnPorch: Three season porch area in square feet #可以分成0-1变量
#%% ScreenPorch: Screen porch area in square feet #可以分成0-1变量
train_data['ScreenPorch_0'] = pd.cut(train_data['ScreenPorch'], bins=[-1,0,100000])
train_data.groupby('ScreenPorch_0')['SalePrice'].mean().plot(kind='bar')

#%% PoolArea: Pool area in square feet #不平衡
train_data['PoolArea'].value_counts()


#%% PoolQC: Pool quality #缺失值很多
train_data['PoolQC'].value_counts()

#%% Fence: Fence quality #缺失值很多
train_data['Fence'].value_counts()
train_data["Fence"] = train_data["Fence"].fillna("None")
train_data.groupby('Fence')['SalePrice'].mean().plot(kind='bar')

#%% MiscFeature: Miscellaneous feature not covered in other categories #缺失值很多
train_data['MiscFeature'].value_counts()


#%% MiscVal: $Value of miscellaneous feature #可以分成0—1变量
train_data['MiscVal'].value_counts()

#%% MoSold: Month Sold #没用
#%% YrSold: Year Sold #没用
#%% SaleType: Type of sale #不平衡，但有用
#%% SaleCondition: Condition of sale #有用
train_data['SaleCondition'].value_counts()
train_data.groupby('SaleCondition')['SalePrice'].mean().plot(kind='bar')

#%% 离群点
train = train.drop(train[(train['GrLivArea']>4000) & (train['SalePrice']<300000)].index)
















