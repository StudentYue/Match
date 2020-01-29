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
import lightgbm as lgb.LGBMRegressor
from sklearn.cross_validation import KFold
from sklearn.model_selection import KFold
#%% 切分
kf = KFold(n_splits=5, shuffle=True)

clfs = [
        GradientBoostingRegressor(),
        RandomForestRegressor(),
        lgb.LGBMRegressor()]


# 对于每一层回归，skf为fold数
for j, clf in enumerate(clfs):
    print('正在用第一层学习器的第%s个'%j)
    blend_test_j = np.zeros((X_test.shape[0], len(skf)))
    # Number of testing data x Number of folds , we will take the mean of the predictions later
    for i, (train_index, cv_index) in enumerate(kf.split(X_train)):
        print 'Fold [%s]' % (i)
            
        # This is the training and validation set
        x_train = X_train.iloc[train_index]
        y_train = Y_train.iloc[train_index]
        x_cv = X_train.iloc[cv_index]
        x_cv = Y_train.iloc[cv_index]
        
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
bclf = lgb.LGBMRegressor()
bclf.fit(blend_train, Y_train)

#%% 4. 再用 bclf 来预测测试集 blend_test，并得到 score：
# Predict now
Y_test_predict = bclf.predict(blend_test)










      
        
        
        
        
        
        
        
        
        
        