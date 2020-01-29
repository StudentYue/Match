p# -*- coding: utf-8 -*-
"""
Created on Mon Feb 18 12:15:31 2019

@author: ZHOU-JC
"""
#%% 
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from xgboost.sklearn import XGBClassifier
from sklearn.metrics import cohen_kappa_score, make_scorer
from sklearn.model_selection import GridSearchCV, cross_val_score, train_test_split
#%% 评价函数 Metric used for this competition 
# (Quadratic Weigthed Kappa aka Quadratic Cohen Kappa Score)
def metric(y1,y2):
    return cohen_kappa_score(y1, y2, weights = 'quadratic')

# Make scorer for scikit-learn
scorer = make_scorer(metric)
#%%
train_df = pd.read_csv('train.csv')
test_df  = pd.read_csv('test.csv')
target = train_df['AdoptionSpeed']

#%% 特征工程
train_df['Type'].value_counts().rename({1:'Dog',2:'Cat'}).plot(kind='barh')
plt.title('猫、狗分类')

# Gender分类
# Age变量
# 

#%% 数据清理
y = train_df['AdoptionSpeed']
x_train = train_df.drop(columns = ['Name', 'RescuerID', 'Description', 'PetID', 'AdoptionSpeed'])
x_test = test_df.drop(columns = ['Name', 'RescuerID', 'Description', 'PetID'])


#%% 用xgboost
#xgb_params = {
#    # 常规参数
#    'nthread':-1, #使用全部CPU
#    # 模型参数 
#    'n_estimatores':2000, #迭代次数
#    'early_stopping_rounds' : 20, #在验证集上，当连续n次迭代，分数没有提高后，提前终止训练          
#    'max_depth':6, 
#    'min_child_weight' : 1,
#    'subsample' : 0.9,
#    'colsample_bytree' : 0.7,
#    # 学习任务参数
#    'learning_rate' : 0.05,
#    'objective' : 'multi:softmax',     #返回概率，用于多分类
#    'eval_metric' : 'mlogloss',        #负对数似然函数（多分类）
#    'alpha' : 1,
#    'lambda' :1
#    }
model_xgb = XGBClassifier(
    # 常规参数
    nthread=-1, #使用全部CPU
    # 模型参数 
    n_estimatores=2000, #迭代次数
    early_stopping_rounds=20, #在验证集上，当连续n次迭代，分数没有提高后，提前终止训练          
    max_depth=6, 
    min_child_weight=1,
    subsample=0.9,
    colsample_bytree=0.7,
    # 学习任务参数
    learning_rate=0.05,
    objective='multi:softmax',     #返回概率，用于多分类
    eval_metric='mlogloss',        #负对数似然函数（多分类）
    reg_alpha=1,
    reg_lambda=1)
model_xgb.fit(x_train, y)

#%%
model_xgb.predict(x_test)
val = cross_val_score(model_xgb, x_train, y, scoring = scorer, cv=3).mean()

#%% 读读json形式
import json

f = open(r'D:\data match\petfinder-adoption-prediction\train_sentiment\00a1f270a.json','r')
sentiment = json.load(f)







