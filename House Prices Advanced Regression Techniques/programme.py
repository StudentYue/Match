# -*- coding: utf-8 -*-
"""
Created on Sun Jan 20 22:21:42 2019

@author: ZHOU-JC
"""


train_data = pd.read_csv(r"C:\Users\ZHOU-JC\Desktop\data match\House Prices Advanced Regression Techniques/train.csv")
test_data = pd.read_csv(r"C:\Users\ZHOU-JC\Desktop\data match\House Prices Advanced Regression Techniques/test.csv")
all_data = train_data.append(test_data).reset_index()

#%% 缺失
a = all_data.isnull().sum().sort_values(ascending = False)
all_data['MSZoning'] = all_data['MSZoning'].value_counts().index[0]
all_data['Exterior1st']
#%% 查看属性

#%% 缺失值
#MSZoning,Exterior1st,Exterior2nd, MasVnrArea, BsmtQual, BsmtFinType1, 
#BsmtFinSF1, TotalBsmtSF, Electrical, BsmtFullBath, KitchenQual, 
#Functional, FireplaceQu, GarageYrBlt, GarageFinish, GarageCars,
#GarageArea, GarageCond, GarageQual, SaleType

all_data['MSZoning'] = all_data['MSZoning'].fillna(all_data['MSZoning'].value_counts().index[0])
all_data['Exterior1st'] = all_data['Exterior1st'].fillna(all_data['Exterior1st'].value_counts().index[0])
all_data['Exterior2nd'] = all_data['Exterior2nd'].fillna(all_data['Exterior2nd'].value_counts().index[0])
all_data['MasVnrArea'] = all_data['MasVnrArea'].fillna(all_data['MasVnrArea'].value_counts().index[0])
all_data['BsmtQual'] = all_data['BsmtQual'].fillna(all_data['BsmtQual'].value_counts().index[0])
all_data['BsmtFinType1'] = all_data['BsmtFinType1'].fillna(all_data['BsmtFinType1'].value_counts().index[0])
all_data['BsmtFinSF1'] = all_data['BsmtFinSF1'].fillna(all_data['BsmtFinSF1'].value_counts().index[0])
all_data['TotalBsmtSF'] = all_data['TotalBsmtSF'].fillna(all_data['TotalBsmtSF'].value_counts().index[0])
all_data['Electrical'] = all_data['Electrical'].fillna(all_data['Electrical'].value_counts().index[0])
all_data['BsmtFullBath'] = all_data['BsmtFullBath'].fillna(all_data['BsmtFullBath'].value_counts().index[0])
all_data['KitchenQual'] = all_data['KitchenQual'].fillna(all_data['KitchenQual'].value_counts().index[0])
all_data['Functional'] = all_data['Functional'].fillna(all_data['Functional'].value_counts().index[0])
all_data['GarageCars'] = all_data['GarageCars'].fillna(all_data['GarageCars'].value_counts().index[0])
all_data['GarageArea'] = all_data['GarageArea'].fillna('none')
all_data['SaleType'] = all_data['SaleType'].fillna('none')


all_data['FireplaceQu'] = all_data['FireplaceQu'].fillna('none')
all_data['GarageYrBlt'] = all_data['GarageYrBlt'].fillna('none')
all_data['GarageFinish'] = all_data['GarageFinish'].fillna('none')
all_data['GarageCond'] = all_data['GarageCond'].fillna('none')
all_data['GarageQual'] = all_data['GarageQual'].fillna('none')




#%% 处理
# 独热编码
train_data['FireplaceQu'].value_counts()
train_data.groupby('FireplaceQu')['SalePrice'].mean().plot(kind='bar')
#LotShape #类别太多
all_data['Neighborhood'] = all_data['LotShape'].map(lambda x:1 if x=='Reg' else 0 )
#Neighborhood #类别太多 街道没办法
#HouseStyle #类别太多 没办法
#RoofStyle  #类别太多
all_data['RoofStyle'] = all_data['RoofStyle'].map(lambda x:2 if x=='Gable' else 1 if x=='Hip' else 0)
#Exterior1st #类别太多 不用
#Exterior2nd #类别太多 不用
#Foundation #类别太多 
all_data['Foundation'] = all_data['Foundation'].map(lambda x:1 if x=='PConc' else 0)
#BsmtFinType1#类别太多
all_data['BsmtFinType1'] = all_data['BsmtFinType1'].map(lambda x:2 if x=='Unf' else 1 if x=='GLQ' else 0)
#Functional#类别太多
all_data['Functional'] = all_data['Functional'].map(lambda x:1 if x=='Typ' else 0)
#SaleType#类别太多
all_data['SaleType'] = all_data['SaleType'].map(lambda x:1 if x==('New' or 'Con') else 0)
#SaleCondition #类别太多
all_data['SaleCondition'] = all_data['SaleCondition'].map(lambda x:1 if x=='Partial' else 0)


cols = ['MSZoning',
        'LotShape',
        'HouseStyle',
        'Neighborhood',
        'RoofStyle',
        'BsmtFinType1',
        'Functional',
        'Foundation',
        'SaleType',
        'SaleCondition',
        'GarageFinish',
        'PavedDrive']
for each in cols:
    all_data = pd.concat([all_data, pd.get_dummies(all_data[each], prefix=each)], axis=1)
    all_data.drop(each, axis=1, inplace=True)

#%% 带有大小的序列型处理
cols = ['ExterQual',
        'BsmtQual',
        'HeatingQC',   #Po只有一个
        'KitchenQual',
        'FireplaceQu', #Po有20个
        'GarageCond', #不太平衡 Po有2个
        'GarageQual']
  
dit = {'Ex':5, 'Gd':4, 'TA':3, 'Fa':2, 'Po':1}
for each in cols:
    all_data[each] = all_data[each].map(dit)

# 数值型，不需要处理
#LotArea
#YearBuilt
#YearRemodAdd
#1stFlrSF
#GrLivArea
#BsmtFullBath
#FullBath
#HalfBath
#KitchenAbvGr
#TotRmsAbvGrd
#Fireplaces
#GarageYrBlt
#GarageCars
#GarageArea
#OverallQual
#OverallCond


#%% 变成0——1变量的 (还未独热编码)
col_0_1= ['MasVnrArea',
          'BsmtFinSF1',
          'TotalBsmtSF',
          'CentralAir', #
          'Electrical', #
          'WoodDeckSF', 
          'OpenPorchSF',
          'EnclosedPorch',
          '3SsnPorch',
          'ScreenPorch']

all_data['CentralAir'] = all_data['CentralAir'].map({'Y':1, 'N':0})
all_data['Electrical'] = all_data['Electrical'].map(lambda x:1 if x == 'SBrkr' else 0 )

for each in col_0_1:
    all_data[each] = pd.cut(all_data[each], bins=[-1,0,10000], labels=[0,1])
    all_data[each] = all_data[each].map({'0':0, '1':1}) #将字符转成数值
    
#%% 丢弃不重要和不平衡的变量
cols = [#不重要的
        'LotConfig','LandContour','Condition1','Condition2','BldgType','MasVnrType','BsmtCond'
        ,'BsmtExposure','BsmtFinType2','BsmtFinSF2','BsmtUnfSF','2ndFlrSF','BedroomAbvGr'
        #类别不平衡的
        ,'GarageType','MoSold','YrSold','FireplaceQu','GarageCond','GarageQual',
        'LotFrontage','Street','Alley','Utilities','LandSlope','RoofMatl','ExterCond'
        ,'Heating','LowQualFinSF','BsmtHalfBath','PoolArea','PoolQC','Fence','MiscFeature','MiscVal',
        #字符型的
        'Exterior1st','Exterior2nd','GarageArea','GarageYrBlt'
        ]    
    
all_data.drop(cols, axis=1, inplace=True)
catecols = list(all_data.select_dtypes(include='object').columns)

#%%
X_train = all_data.iloc[:1460,:].drop('SalePrice', axis=1)
Y_train = train_data['SalePrice']
    
X_test  = all_data.iloc[1460:,:].drop('SalePrice', axis=1)
    
    
    
#%% 
from sklearn.model_selection import GroupKFold
from sklearn.preprocessing import minmax_scale
import lightgbm as lgb

params={'learning_rate': 0.05,
        'objective':'mae',
        'metric':'mae',
        'num_leaves': 31,
        'verbose': 0,
        'random_state':42,
        'bagging_fraction': 0.8,
        'feature_fraction': 0.8
       }
    
reg = lgb.LGBMRegressor(**params, n_estimators=200)
reg.fit(X_train, Y_train)
result = reg.predict(X_test)
result = pd.DataFrame(result, index=X_test.Id, columns=['SalePrice'])
result = result.reset_index()
result.to_csv(r"C:\Users\ZHOU-JC\Desktop\data match\House Prices Advanced Regression Techniques/result.csv", index=False)
    