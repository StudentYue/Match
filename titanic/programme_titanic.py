# -*- coding: utf-8 -*-
"""
Created on Sat Jan 12 14:14:31 2019

@author: ZHOU-JC
"""
# =============================================================================
# 日志
#去掉SibSp和Parch后0.78947,排名3596
# =============================================================================



#%%
import pandas as pd 
import numpy as np

train_data = pd.read_csv(r'C:\Users\ZHOU-JC\Desktop\数据挖掘项目\titanic\train.csv', engine='python')
test_data  = pd.read_csv(r'C:\Users\ZHOU-JC\Desktop\数据挖掘项目\titanic\test.csv', engine='python')
#%%画图
from pylab import  * #识别中文字体用
mpl.rcParams['font.sans-serif'] = ['SimHei']   #画图识别中文

#Survived_0 = train_data.Pclass[train_data.Survived == 0].value_counts()
#Survived_1 = train_data.Pclass[train_data.Survived == 1].value_counts()
#df=pd.DataFrame({u'获救':Survived_1, u'未获救':Survived_0})
#df.plot(kind='bar', stacked=True,color=['lightcoral','lightgreen']) #为获救赋予绿色，未获救赋予红色
#plt.title(u"各乘客等级的获救情况")
#plt.xlabel(u"乘客等级")
#plt.ylabel(u"人数")
#plt.show()

Survived_0 = train_data.Parch[train_data.Survived == 0].value_counts()
Survived_1 = train_data.Parch[train_data.Survived == 1].value_counts()
df=pd.DataFrame({u'获救':Survived_1, u'未获救':Survived_0})
df.plot(kind='bar', stacked=True,color=['lightcoral','lightgreen']) #为获救赋予绿色，未获救赋予红色
plt.title(u"各乘客等级的获救情况")
plt.xlabel(u"乘客等级")
plt.ylabel(u"人数")
plt.show()
#%% 数据基本描述、初步处理
statis_train = train_data.describe() 
statis_test  = test_data.describe()
null_train = train_data.isnull().sum()
null_test  = test_data.isnull().sum()
null = train_data.isnull().sum()
#对Name提取信息
import re
regx = re.compile('(.*, )|(\..*)')
title=[]

for name in train_data.Name.values:
    title.append(re.sub(regx,'',name))
train_data['Title']=title
train_data['Title']=list(map(lambda x:"rare" if x not in ['Mr','Miss', 'Mrs', 'Master'] else x, train_data['Title'].tolist()))

title=[]
for name in test_data.Name.values:
    title.append(re.sub(regx,'',name))
test_data['Title']=title
test_data['Title']=list(map(lambda x:"rare" if x not in ['Mr','Miss', 'Mrs', 'Master'] else x, test_data['Title'].tolist()))



#去掉年龄属性
#train_data.drop= train_data[train_data['Age'].notnull()]   
train_data.drop(['Age'], axis=1, inplace=True)
test_data.drop(['Age'], axis=1, inplace=True)
                 
#去掉缺失值超过15%的特征,填补缺失值
train_data.drop(['Cabin'], axis=1, inplace = True)
test_data.drop(['Cabin'], axis=1, inplace = True)
train_data['Embarked'] = train_data['Embarked'].fillna(train_data['Embarked'].value_counts().index[0])
test_data['Fare']  = test_data['Fare'].fillna(test_data['Fare'].mean())

#增加新的属性
train_data['familynum'] = train_data['SibSp'] + train_data['Parch']
test_data['familynum'] = test_data['SibSp'] + test_data['Parch']
labels = [0, 1,2]
bins = [-0.1,0.9,3,test_data['familynum'].max()+1]
train_data['familynum'] = pd.cut(train_data.familynum, bins, labels = labels)
test_data['familynum'] = pd.cut(test_data.familynum, bins, labels = labels)
#对SibSp和Parch进行分箱
labels = [0, 1,2]
bins = [-0.1,0.9,2,test_data['SibSp'].max()+1]
train_data['SibSp'] = pd.cut(train_data.SibSp, bins, labels = labels)
test_data['SibSp']  = pd.cut(test_data.SibSp, bins, labels = labels)

labels = [0, 1,2]
bins = [-0.1,0.9,2,test_data['Parch'].max()+1]
train_data['Parch'] = pd.cut(train_data.Parch, bins, labels = labels)
test_data['Parch']  = pd.cut(test_data.Parch, bins, labels = labels)


##区分数值、属性特征
#numurical_feature = [] 
#category_feature = []
#for each in X_train.columns:
#    if X_train[each].dtype == 'object':
#        category_feature.append(each)
#    else:
#        numurical_feature.append(each)
#独热编码
from sklearn.preprocessing import OneHotEncoder
from sklearn.preprocessing import LabelEncoder

onehotencoder = OneHotEncoder(sparse=False) #不用稀疏正则化
onehotencoder.fit(train_data['Sex'].values.reshape(-1,1))
#对“Sex”属性的独热编码
sex_feature = onehotencoder.transform(train_data['Sex'].values.reshape(-1,1))
sex = pd.DataFrame(sex_feature, columns=['sex_1','sex_2'], index=train_data.index)
train_data = pd.merge(train_data, sex, left_index=True, right_index=True)
train_data = train_data.drop(['Sex'], axis=1)

sex_feature = onehotencoder.transform(test_data['Sex'].values.reshape(-1,1))
sex = pd.DataFrame(sex_feature, columns=['sex_1','sex_2'], index=test_data.index)
test_data = pd.merge(test_data, sex, left_index=True, right_index=True)
test_data = test_data.drop(['Sex'], axis=1)

#对“Embarked”属性的独热编码
a = pd.get_dummies(train_data['Embarked'], prefix='Embarked', )
train_data = pd.merge(train_data, a, left_index=True, right_index=True)
train_data.drop(['Embarked'], axis=1, inplace=True)
a = pd.get_dummies(test_data['Embarked'], prefix='Embarked')
test_data  = pd.merge(test_data, a, left_index=True, right_index=True)
test_data.drop(['Embarked'], axis=1, inplace=True)

#对"Pclass"属性的独热编码
a = pd.get_dummies(train_data['Pclass'], prefix='Pclass', )
train_data = pd.merge(train_data, a, left_index=True, right_index=True)
train_data.drop(['Pclass'], axis=1, inplace=True)
a = pd.get_dummies(test_data['Pclass'], prefix='Pclass')
test_data  = pd.merge(test_data, a, left_index=True, right_index=True)
test_data.drop(['Pclass'], axis=1, inplace=True)

#对"Title"属性的独热编码
a = pd.get_dummies(train_data['Title'], prefix='Title', )
train_data = pd.merge(train_data, a, left_index=True, right_index=True)
train_data.drop(['Title'], axis=1, inplace=True)
a = pd.get_dummies(test_data['Title'], prefix='Title')
test_data  = pd.merge(test_data, a, left_index=True, right_index=True)
test_data.drop(['Title'], axis=1, inplace=True)
#对"familynum"属性的独热编码
a = pd.get_dummies(train_data['familynum'], prefix='familynum', )
train_data = pd.merge(train_data, a, left_index=True, right_index=True)
train_data.drop(['familynum'], axis=1, inplace=True)
a = pd.get_dummies(test_data['familynum'], prefix='familynum')
test_data  = pd.merge(test_data, a, left_index=True, right_index=True)
test_data.drop(['familynum'], axis=1, inplace=True)
##对"SibSp"和"Parch"属性的独热编码
#a = pd.get_dummies(train_data['SibSp'], prefix='SibSp', )
#train_data = pd.merge(train_data, a, left_index=True, right_index=True)
#
#a = pd.get_dummies(test_data['SibSp'], prefix='SibSp')
#test_data  = pd.merge(test_data, a, left_index=True, right_index=True)
#
#a = pd.get_dummies(train_data['Parch'], prefix='Parch', )
#train_data = pd.merge(train_data, a, left_index=True, right_index=True)
#
#a = pd.get_dummies(test_data['Parch'], prefix='Parch')
#test_data  = pd.merge(test_data, a, left_index=True, right_index=True)

#去掉个人觉得影响不大的属性【‘ticket’】
train_data.drop(['Ticket'], axis=1, inplace=True)
test_data.drop(['Ticket'], axis=1, inplace=True)

#提取X和Y
X_train = train_data.drop(['Survived','PassengerId','Name','SibSp','Parch'], axis=1)
X_test  = test_data.drop(['PassengerId','Name','SibSp','Parch'], axis=1)
Y_train = train_data['Survived']    
#%% 特征选择
#区分数值、属性特征
numurical_feature = [] 
category_feature = []
for each in X_train.columns:
    if X_train[each].dtype == 'object':
        category_feature.append(each)
    else:
        numurical_feature.append(each)
        
#%% 过滤(filter)特征选择
#Pearson相关系数  
Feature_Pearson = train_data.corr()['Survived'].sort_values(ascending=False)
#卡方验证
from sklearn.feature_selection import SelectKBest
from sklearn.feature_selection import chi2

model1 = SelectKBest(chi2, k=4)#选择k个最佳特征
model1.fit_transform(train_data[numurical_feature].values.reshape(-1,len(numurical_feature)), train_data['Survived'].values.reshape(-1,1))
Feature_chi2 = pd.DataFrame(model1.scores_, index=numurical_feature)

#互信息和最大信息系数

#距离相关系数

#方差选择法

#%%包装(wrapper)特征选择
#前向搜索


#后向搜索


#递归特征消除法
from sklearn.feature_selection import RFE
from sklearn.tree import DecisionTreeClassifier

rfe = RFE(estimator=DecisionTreeClassifier(), n_features_to_select=4)
rfe.fit_transform(X_train[numurical_feature].values,Y_train.values)
Feature_RFE = X_train[numurical_feature].columns[rfe.get_support()] #查看满足条件的特征

#%%嵌入(Embedded)特征选择
#L1正则化挑选参数,使用L1时一定要对数据进行归一化
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import Lasso,LassoCV,LassoLarsCV   # Lasso回归,LassoCV交叉验证实现alpha的选取，LassoLarsCV基于最小角回归交叉验证实现alpha的选取
#数据标准化
standard = StandardScaler()
standard.fit(X_train[numurical_feature])
X_train_standar = standard.transform(X_train[numurical_feature])

model = LassoCV()
model.fit(X_train_standar, Y_train.values)

Feature_lasso = pd.DataFrame(model.coef_, index = numurical_feature)
print("最佳的alpha:",model.alpha_)

#%%L2正则化挑选参数
from sklearn.linear_model import RidgeCV
model = RidgeCV()
model.fit(X_train_standar, Y_train.values)
Feature_ridgfe = pd.DataFrame(model.coef_, index = numurical_feature)
print("最佳的alpha:",model.alpha_)





#%% 开始进行训练
#xgb
import xgboost as xgb
import modelfit
from xgboost.sklearn import XGBClassifier
from sklearn.model_selection import GridSearchCV

xgb1 = XGBClassifier(
 learning_rate =0.1,
 n_estimators=24,
 max_depth=9,
 min_child_weight=1,
 gamma=0,
 subsample=0.8,
 colsample_bytree=0.9,
 objective= 'binary:logistic',
 eval_metric='error',
 nthread=4,
 scale_pos_weight=1,
 seed=27,
 reg_alpha=0.005,
 reg_lambda=0.001)
xgb1.fit(X_train, Y_train)

##利用xgb.cv调参，确定树的个数n_estimators
#xgb1_param = xgb1.get_xgb_params()
#xgb_train = xgb.DMatrix(train_data[numurical_feature].values, label=train_data['Survived'].values)
#cvresult = xgb.cv(xgb1_param, xgb_train, num_boost_round=xgb1.get_params()['n_estimators'], 
#                  nfold=5, metrics='auc', early_stopping_rounds=50)
#xgb1.set_params(n_estimators=cvresult.shape[0])
#
##利用grid search确定max_depth 和 min_weight
#param_test1 = {'max_depth':range(3,10,2),
#               'min_child_weight':range(1,6,2)}
#gsearch1 = GridSearchCV(estimator = XGBClassifier(learning_rate =0.1,
#           n_estimators=12, max_depth=5,min_child_weight=1, gamma=0, 
#           subsample=0.8,colsample_bytree=0.8,objective= 'binary:logistic',
#           nthread=4,scale_pos_weight=1, seed=27), 
#            param_grid = param_test1, n_jobs=4,iid=False, cv=5)
#gsearch1.fit(X_train, Y_train)
#gsearch1.best_params_
#gsearch1.best_score_
#
##利用grid search确定gamma
#param_test3 = {'gamma':[i/10.0 for i in range(0,5)]}
#gsearch3 = GridSearchCV(estimator = XGBClassifier(
#        learning_rate =0.1, n_estimators=12, max_depth=9, 
#        min_child_weight=1, gamma=0, subsample=0.8, colsample_bytree=0.8, 
#        objective= 'binary:logistic', nthread=4, scale_pos_weight=1,seed=27), 
#        param_grid = param_test3, scoring='roc_auc',n_jobs=4,iid=False, cv=5)
#gsearch3.fit(X_train, Y_train)
#gsearch3.best_params_, gsearch3.best_score_
#
##调整subsample 和 colsample_bytree 参数
#param_test4 = {'subsample':[i/10.0 for i in range(6,10)],
#               'colsample_bytree':[i/10.0 for i in range(6,10)]}
#
#gsearch4 = GridSearchCV(estimator = XGBClassifier( 
#        learning_rate =0.1, n_estimators=12, max_depth=9, 
#        min_child_weight=1, gamma=0.4, subsample=0.8, colsample_bytree=0.8, 
#        objective= 'binary:logistic', nthread=4, scale_pos_weight=1,seed=27),
#        param_grid = param_test4, scoring='roc_auc',n_jobs=4,iid=False, cv=5)
#
#gsearch4.fit(X_train, Y_train)
#gsearch4.best_params_, gsearch4.best_score_
#
##正则化参数调优
#param_test6 = {'reg_alpha':[0, 0.001, 0.005, 0.01, 0.05],
#               'reg_lambda':[0, 0.001, 0.005, 0.01, 0.05]}
#gsearch6 = GridSearchCV(estimator = XGBClassifier( 
#        learning_rate =0.1, n_estimators=12, max_depth=9, 
#        min_child_weight=1, gamma=0.4, subsample=0.8, colsample_bytree=0.8, 
#        objective= 'binary:logistic', nthread=4, scale_pos_weight=1,seed=27),
#        param_grid = param_test6, scoring='roc_auc',n_jobs=4,iid=False, cv=5)
#
#gsearch6.fit(X_train, Y_train)
#gsearch6.best_params_, gsearch6.best_score_


#%%保存结果到csv格式
resultt_test = xgb1.predict(X_test)
resultt_train = xgb1.predict(X_train)
precision = resultt_train == Y_train.values
precision = precision.sum()/Y_train.shape[0]

result=pd.DataFrame(resultt_test)
result=pd.concat((test_data['PassengerId'], result), axis=1)
result = result.rename(columns={0:'Survived'})
result.to_csv(r'C:\Users\ZHOU-JC\Desktop\数据挖掘项目\titanic\result.csv', index=False)


















