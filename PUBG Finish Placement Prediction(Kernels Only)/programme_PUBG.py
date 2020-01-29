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
def fillInf(df, val):
    numcols = df.select_dtypes(include='number').columns
    cols = numcols[numcols != 'winPlacePerc']
    df[df == np.Inf] = np.NaN
    df[df == np.NINF] = np.NaN
    for c in cols: df[c].fillna(val, inplace=True)
def reduce_mem_usage(df):
    """ iterate through all the columns of a dataframe and modify the data type
        to reduce memory usage.
    """
    start_mem = df.memory_usage().sum() / 1024**2
    
    for col in df.columns:
        col_type = df[col].dtype
        
        if col_type != object:
            c_min = df[col].min()
            c_max = df[col].max()
            if str(col_type)[:3] == 'int':
                if c_min > np.iinfo(np.int8).min and c_max < np.iinfo(np.int8).max:
                    df[col] = df[col].astype(np.int8)
                elif c_min > np.iinfo(np.int16).min and c_max < np.iinfo(np.int16).max:
                    df[col] = df[col].astype(np.int16)
                elif c_min > np.iinfo(np.int32).min and c_max < np.iinfo(np.int32).max:
                    df[col] = df[col].astype(np.int32)
                elif c_min > np.iinfo(np.int64).min and c_max < np.iinfo(np.int64).max:
                    df[col] = df[col].astype(np.int64)  
            else:
                #if c_min > np.finfo(np.float16).min and c_max < np.finfo(np.float16).max:
                #    df[col] = df[col].astype(np.float16)
                #el
                if c_min > np.finfo(np.float32).min and c_max < np.finfo(np.float32).max:
                    df[col] = df[col].astype(np.float32)
                else:
                    df[col] = df[col].astype(np.float64)
        #else:
            #df[col] = df[col].astype('category')

    end_mem = df.memory_usage().sum() / 1024**2
    print('Memory usage of dataframe is {:.2f} MB --> {:.2f} MB (Decreased by {:.1f}%)'.format(
        start_mem, end_mem, 100 * (start_mem - end_mem) / start_mem))
    return df


#%%
train_data  = pd.read_csv(r'C:\Users\ZHOU-JC\Desktop\data match\PUBG Finish Placement Prediction(Kernels Only)\train_V2.csv', nrows=500000)
#test_data   = pd.read_csv(r'C:\Users\ZHOU-JC\Desktop\data match\PUBG Finish Placement Prediction(Kernels Only)\test_V2.csv', nrows=1000000)
# 删掉缺失值
null_cnt = train_data.isnull().sum().sort_values()
print('null count:', null_cnt[null_cnt > 0])
# dropna
train_data.dropna(inplace=True)

train_data,test_data = train_test_split(train_data, test_size=0.2, random_state=42)
train_data = reduce_mem_usage(train_data)
test_data = reduce_mem_usage(test_data)
real = test_data['winPlacePerc']
test_data['winPlacePerc'] = np.nan

test_data.drop(['winPlacePerc'], axis=1, inplace=True)

#%%
e = train_data.groupby(['matchId','groupId']).mean().reset_index()
d = pd.merge(test_data,  e[['matchId','groupId','winPlacePerc']], how='left')
test_data = d
#%%
 
all_data = train_data.append(test_data, sort=False).reset_index(drop=True)
# rank as percent
match = all_data.groupby('matchId')
all_data['killPlacePerc'] = match['kills'].rank(pct=True).values
all_data['walkDistancePerc'] = match['walkDistance'].rank(pct=True).values

# 增加属性：总距离
all_data['distance'] = all_data['swimDistance'] +all_data['rideDistance']
# 增加属性：engage(助攻+自己击倒)
all_data['engage'] = all_data['kills'] + all_data['assists'] 
# 增加属性：总用药数
all_data['healthItems'] = all_data['heals'] + all_data['boosts']

#%%删除不重要的
all_data.drop((['headshotKills', 'teamKills', 'roadKills', 'vehicleDestroys']), axis=1, inplace=True)
all_data.drop((['rideDistance', 'swimDistance', 'matchDuration']), axis=1, inplace=True)
all_data.drop((['rankPoints', 'killPoints', 'winPoints']), axis=1, inplace=True)

#%% 按match和group分组预测
match = all_data.groupby('matchId')
group = all_data.groupby(['matchId','groupId','matchType'])

# target目标特征
agg_col = list(all_data.columns)
exclude_agg_col = ['Id','matchId','groupId','matchType','maxPlace','numGroups','winPlacePerc']
for c in exclude_agg_col:
    agg_col.remove(c)

sum_col = ['kills', 'killPlace', 'damageDealt', 'walkDistance', 'healthItems']


#%% 每场比赛的人数，每支队伍的人数
# match sum、max , group sum 关于sum_col
match_data=pd.concat([
        match.size().to_frame('m.players'),
        match[sum_col].sum().rename(columns=lambda s:'m.sum.'+s),
        match[sum_col].max().rename(columns=lambda s:'m.max.'+s)
        ], axis=1).reset_index() #reset_inde 这波操作会令index变成顺序数字，而columns会多出个matchId
#merge inner join 的方法，使用内链接                
match_data = pd.merge(match_data, group[sum_col].sum().rename(columns=lambda s:'sum.'+s).reset_index())
match_data = reduce_mem_usage(match_data)

print(match_data.shape)

#%% ranking of kills and killPlace in each match


#%% group mean, max, min
# 假如是单人的话mean, max, min一样的
all_data = pd.concat([
        group.size().to_frame('players'),#队伍里有多少人
        group.mean(),
        group[agg_col].max().rename(columns=lambda s:'max.'+s),
        group[agg_col].min().rename(columns=lambda s:'min.'+s),
        ], axis=1).reset_index()

    
all_data = reduce_mem_usage(all_data)

print(all_data.shape)


#%% 融合特征
numcols = list(all_data.select_dtypes(include='number').columns)
numcols.remove('winPlacePerc')

#%% 关键一步，all_data包含队伍的信息，match_data包含每场比赛的总数和最大值
all_data = pd.merge(all_data, match_data)
del match_data

all_data['enemy.players'] = all_data['m.players'] - all_data['players']
for c in sum_col:
#敌人平均的kills,killPlace,damageDealt,walkDistance,healthItemns
    all_data['enemy.' + c] = (all_data['m.sum.' + c] - all_data['sum.' + c]) / all_data['enemy.players']
    #all_data['p.sum_msum.' + c] = all_data['sum.' + c] / all_data['m.sum.' + c]
#团队最大值占比赛总值的多少
    all_data['p.max_msum.' + c] = all_data['max.' + c] / all_data['m.sum.' + c]
#团队最大值占比赛最大值的多少
    all_data['p.max_mmax.' + c] = all_data['max.' + c] / all_data['m.max.' + c]
#丢掉比赛总值和比赛的最大值，因为这对预测没有帮助
    all_data.drop(['m.sum.' + c, 'm.max.' + c], axis=1, inplace=True)

fillInf(all_data, 0)    
print(all_data.shape)

#%% match rank 将一些数值属性转换为白比分排名

match = all_data.groupby('matchId')
#用百分比会提高精准率的吗
matchRank = match[numcols].rank(pct=True). rename(columns=lambda s:'rank.'+s)
all_data = reduce_mem_usage(pd.concat([all_data, matchRank], axis=1))
rank_col = matchRank.columns
del matchRank


# instead of rank(pct=True, method='dense')
match = all_data.groupby('matchId')
matchRank = match[rank_col].max().rename(columns=lambda s: 'max.' + s).reset_index()
all_data = pd.merge(all_data, matchRank) #这步all_data是DataFrame形式
#队伍排名占比赛最高排名的比例
for c in numcols:
    all_data['rank.' + c] = all_data['rank.' + c] / all_data['max.rank.' + c]
    all_data.drop(['max.rank.' + c], axis=1, inplace=True)
del matchRank

print(all_data.shape) 
# 经检查 这里并没有nan值

#%% killPlace rank of group and kills

#%% delete feature
# 删除只有一个属性的项
# 能否删除属性比例不平衡的项
constant_column = [col for col in all_data.columns if all_data[col].nunique() == 1]
print('drop columns:', constant_column)
all_data.drop(constant_column, axis=1, inplace=True)

#%% match type
mapper = lambda x: 'solo' if ('solo' in x) else 'duo' if ('duo' in x) or ('crash' in x) else 'squad'
all_data['matchTypeCat'] = all_data['matchType'].map(mapper)

print(all_data['matchTypeCat'].value_counts())


#%% encode 编码
cols = [col for col in all_data.columns if col not in ['Id', 'matchId', 'groupId']]
for i,t in all_data.loc[:,cols].dtypes.iteritems():
    if t == object:
        all_data[i] = pd.factorize(all_data[i])[0]

all_data = reduce_mem_usage(all_data)
all_data.head()
# 只对了matchType和matchTypeCat 编码 

#%% Predict
# 组装X_train,Y_train,X_test
# all_data有特征列matchId,groupId, 但假如一个group里面既有null又有非空值，则怎么算呢
X_train = all_data[all_data['winPlacePerc'].notnull()].reset_index(drop=True)
X_test = all_data[all_data['winPlacePerc'].isnull()].drop(['winPlacePerc'], axis=1).reset_index(drop=True)

Y_train = X_train.pop('winPlacePerc')
X_test_grp = X_test[['matchId','groupId']].copy()
#train_matchId = X_train['matchId']

# drop matchId,groupId
X_train.drop(['matchId','groupId'], axis=1, inplace=True)
X_test.drop(['matchId','groupId'], axis=1, inplace=True)

print(X_train.shape, X_test.shape)

#%% xgboost
X_test = all_data[all_data['winPlacePerc'].isnull()].drop(['winPlacePerc'], axis=1).reset_index(drop=True)
X_test.drop(['matchId','groupId'], axis=1, inplace=True)

import xgboost as xgb
import lightgbm as lgb

params={'learning_rate': 0.1,    #学习速率
        'n_estimators': 1300, 
        'num_leaves': 50, 
        'max_depth': 5,          #叶的最大深度
        'min_data_in_leaf': 8,  
        'metric':'rmse',          #目标函数
        'objective': 'regression', 
        'verbose': 0,
        'bagging_fraction': 0.5,    #每次迭代用的数据比例
        'feature_fraction': 0.6,     #每次迭代用的特征比例
        'reg_alpha': 1,
        'reg_lambda': 0.01,
        'random_state': 8}
        


mts = []
fis = []
# 
#pred = np.zeros(X_test.shape[0])
#for mt in X_train['matchTypeCat'].unique():
#    idx = X_train[X_train['matchTypeCat'] == mt].index
#    reg = lgb.LGBMRegressor(**params)
#    reg.fit(X_train.loc[idx], Y_train.loc[idx])
#    
#    idx = X_test[X_test['matchTypeCat'] == mt].index
#    pred[idx] = reg.predict(X_test.loc[idx])
#    mts.append(mt)
#    fis.append(reg.feature_importances_)
    
# 调参
for mt in X_train['matchTypeCat'].unique():
    idx = X_train[X_train['matchTypeCat'] == mt].index    
    parameters = {
              'num_leaves': range(50,150,50),
              'max_depth': range(5,20,5),
              }
    reg = lgb.LGBMRegressor(**params)
    gsearch = GridSearchCV(reg, param_grid=parameters, scoring='neg_mean_absolute_error', 
                           cv=3, n_jobs=-1)
    gsearch.fit(X_train.loc[idx], Y_train.loc[idx])
    gsearch.best_params_
    print('%s最佳参数为%s'%(mt, gsearch.best_params_))
    mts.append(mt)



    
    

#model_xgb.fit(X_train, Y_train)
#result = model_xgb.predict(X_test)
#%% 令同一场比赛同一支队伍的成绩相等
X_test_grp['_nofit.winPlacePerc'] = pred #是按groupId预测的，所以不会有重复的group

group = X_test_grp.groupby(['matchId'])
X_test_grp['winPlacePerc'] = pred
X_test_grp['_rank.winPlacePerc'] = group['winPlacePerc'].rank(method='min')
X_test = pd.concat([X_test, X_test_grp], axis=1)

sub_match = X_test_grp[['matchId','_rank.winPlacePerc']].groupby(['matchId'])
sub_group = group.count().reset_index()['matchId'].to_frame()
#注意count()和size()的区别
X_test = pd.merge(X_test, sub_group)

# 
fullgroup = (X_test['numGroups'] == X_test['maxPlace'])

# full group (201366) --> calculate from rank
subset = X_test.loc[fullgroup]

#%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
# 和解？
#X_test.loc[fullgroup, 'winPlacePerc'] = (subset['_rank.winPlacePerc'].values - 1) / (subset['maxPlace'].values - 1)
#
## not full group (684872) --> align with maxPlace
#subset = X_test.loc[~fullgroup]
#gap = 1.0 / (subset['maxPlace'].values - 1)
#new_perc = np.around(subset['winPlacePerc'].values / gap) * gap  # half&up
#X_test.loc[~fullgroup, 'winPlacePerc'] = new_perc
#
#X_test['winPlacePerc'] = X_test['winPlacePerc'].clip(lower=0,upper=1)
#
##
#X_test.loc[~fullgroup, '_pred.winPlace'] = np.around(X_test.loc[~fullgroup, 'winPlacePerc'].values / gap) + 1
#X_test.loc[~fullgroup & (X_test['matchId'] == '000b598b79aa5e'),
#           ['matchId','groupId','winPlacePerc','maxPlace','numGroups','_pred.winPlace','_rank.winPlacePerc']
#          ].sort_values(['matchId','_pred.winPlace','_rank.winPlacePerc'])
#
## edge cases
#X_test.loc[X_test['maxPlace'] == 0, 'winPlacePerc'] = 0
#X_test.loc[X_test['maxPlace'] == 1, 'winPlacePerc'] = 1  # nothing
#X_test.loc[(X_test['maxPlace'] > 1) & (X_test['numGroups'] == 1), 'winPlacePerc'] = 0
#%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

X_test['winPlacePerc'].describe()

#%% 
b = X_test[['matchId','groupId','winPlacePerc']]

test_data['winPlacePerc_0'] = test_data['winPlacePerc'].fillna(0)
test_data.drop('winPlacePerc', axis=1, inplace=True)
result = pd.merge(test_data, b, how='left')

result['winPlacePerc_0'] = result['winPlacePerc_0'].fillna(0)
result['winPlacePerc']   = result['winPlacePerc'].fillna(0)
result['a'] = result['winPlacePerc_0'] + result['winPlacePerc']

precision = abs(result['a'].values - real.values).mean()





















