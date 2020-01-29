# -*- coding: utf-8 -*-
"""
Created on Tue Jan 15 19:41:21 2019

@author: ZHOU-JC
"""
#%%
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import itertools
import gc
import os
import sys
import reduce_mem_usage

sns.set_style('darkgrid')
sns.set_palette('bone')


#pd.options.display.float_format = '{:.5g}'.format
pd.options.display.float_format = '{:,.2f}'.format


#print(os.listdir("../input"))
#%%
def toTapleList(list1,list2):
    return list(itertools.product(list1,list2))
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


#%% 加载数据
train_data = pd.read_csv(r'C:\Users\ZHOU-JC\Desktop\data match\PUBG Finish Placement Prediction(Kernels Only)\train_V2.csv')
train_data = reduce_mem_usage(train_data)
test_data = pd.read_csv(r'C:\Users\ZHOU-JC\Desktop\data match\PUBG Finish Placement Prediction(Kernels Only)\test_V2.csv')
test_data = reduce_mem_usage(test_data)
X = train_data


#%% 删除缺失值
train_data.dropna(inplace=True)



#%% 查看基础信息
#a = train_data.describe().drop('count').T



#%% Id, groupId, matchId
#for each in ['Id', 'groupId', 'matchId']:
#    print(each,'unique count:',train_data[each].nunique())


#%% Id, groupId, matchId
#for c in ['Id', 'groupId', 'matchId']:
#    print('unique[%s]count:'%c, train_data[c].nunique()) #nunique为独特的个数 
#    
#%% matchType

fig, ax = plt.subplots(1, 2, figsize=(12, 4))
a = train_data.groupby('matchId')['matchType'].first().value_counts().plot(kind='bar', ax=ax[0])
mapper = lambda x:'solo' if ('solo' in x) else 'duo' if ('duo' in x) or ('crash' in x) else 'squad'
train_data['matchType'] = train_data['matchType'].apply(mapper)
train_data.groupby('matchId')['matchType'].first().value_counts().plot(kind='bar',ax=ax[1])

#%% maxPlace, numGroups
#query后面只支持string形式的值，而‘age’==24返回的是一个bool类型，结果不是true就是false，所以需要进行如下操作，才可返回正确结果
for q in ['numGroups == maxPlace', 'numGroups != maxPlace']:
    print(q,':',len(train_data.query(q)))


#%%
cols = ['numGroups', 'maxPlace']
desc1 = train_data.groupby('matchType')[cols].describe()[toTapleList(cols,['min','mean','max'])]
group = train_data.groupby(['matchType','matchId','groupId']).count()



#%% players
#为什么单排会出现有队友，双排会出现多于两个人
#group = train_data.groupby(['matchId','groupId','matchType'])['Id'].count().to_frame('players').reset_index()
#group.loc[group['players'] > 4, 'players'] = '5+'
#group['players'] = group['players'].astype(str)
#
#fig, ax = plt.subplots(1, 3, figsize=(16, 3))
#for mt, ax in zip(['solo','duo','squad'], ax.ravel()):
#    ax.set_xlabel(mt)
#    group[group['matchType'] == mt]['players'].value_counts().sort_index().plot.bar(ax=ax)



#%% matchDuration 比赛时间
fig, ax = plt.subplots(1, 2, figsize = (12, 4))
train_data['matchDuration'].plot(kind='hist', bins=50, ax=ax[0])
train_data.query('matchDuration >= 1400 & matchDuration <= 1800')['matchDuration'].hist(bins=50, ax=ax[1])

a = train_data[train_data['matchDuration'] == train_data['matchDuration'].max()]
# same match is same duration
(train_data.groupby('matchId')['matchDuration'].nunique()>1).any()

#%% boosts, heals 
train_data.loc[train_data['boosts'] > 0, 'boosts']='1+'
train_data['boosts'] = train_data['boosts'].astype(str)
train_data['boosts'].value_counts().plot(kind='bar')
train_data.groupby('boosts')['winPlacePerc'].mean().plot(kind='bar')

train_data.loc[train_data['heals'] > 1, 'heals']='1+'
train_data['heals'] = train_data['heals'].astype(str)
train_data['heals'].value_counts().plot(kind='bar')
train_data.groupby('heals')['winPlacePerc'].mean().plot(kind='bar')
#%% rivives 救人数
#单排玩家救人数=0
col = 'revives'
sub = train_data.loc[~train_data['matchType'].str.contains('solo'),['winPlacePerc', col]].copy()
sub.loc[sub['revives'] > 0, 'revives']='0+'
sub['revives'] = sub['revives'].astype(str)
sub['revives'].value_counts().plot(kind='bar')
sub.groupby('revives')['winPlacePerc'].mean().plot(kind='bar')

#%% killPlace 和 kills
# 这两参数很重要
# 这两参数与matchType影响不大
train_data['killPlace'].plot(kind='hist', bins=50)
col = 'killPlace'
sub = train_data[['winPlacePerc',col]].copy()
sub[col] = pd.cut(sub[col], 10, right=False)
sub.groupby(col)['winPlacePerc'].mean().plot(kind='bar')

sub = train_data.loc[train_data['kills']<5,['kills','winPlacePerc']].copy()
sub.plot(kind='hist', bins=5)
sub.groupby('kills')['winPlacePerc'].mean().plot(kind='bar')

#%% killStreaks,DBNOs
#killStreak为连杀数，DBNOs为击倒数
sub = train_data.copy()
sub.loc[sub['killStreaks']>1, 'killStreaks'] = '1+'
sub[col].astype(str)
sub['killStreaks'].value_counts().plot(kind='bar')
sub.groupby('killStreaks')['winPlacePerc'].mean().plot(kind='bar')

sub = train_data.copy()
sub.loc[sub['DBNOs']>0, 'DBNOs'] = '0+'
sub[col].astype(str)
sub['DBNOs'].value_counts().plot(kind='bar')
sub.groupby('DBNOs')['winPlacePerc'].mean().plot(kind='bar')
    
#%% headshotKills,roadKills,teamKills
#headshotKills为爆头数,roadKills为在车上的杀人数,teamKills为杀队友数
#heakshotKills可以用一下，其他类别非常不平衡
fig, ax = plt.subplots(3, 2, figsize=(16,12))
cols = ['headshotKills', 'roadKills', 'teamKills']
for col, ax in zip(cols, ax):
    sub = train_data[['winPlacePerc', col]].copy()
    sub.loc[sub[col] > 0, col] = '0+'
    sub[col] = sub[col].astype(str)
    sub.groupby(col).mean()['winPlacePerc'].plot(kind='bar', ax=ax[0])
    sub[col].value_counts().plot(kind='bar', ax=ax[1])
    
#%% assists
#助攻数
sub = train_data.copy()
sub.loc[sub['assists'] >= 5, 'assists'] = '5+'
sub['assists'] = sub['assists'].astype(str)
sub['assists'].plot(kind='bar')






#%% longestKill
fig, ax = plt.subplots(1,2,figsize=(16,3))
col = 'longestKill'
sub = train_data[['winPlacePerc', col]].copy()
sub[col] = pd.cut(sub[col], 6)
sub.groupby(col).mean()['winPlacePerc'].plot(kind='bar', ax=ax[0])
train_data[col].plot(kind='hist', bins=20, ax=ax[1])

#%% damageDealt
#很奇怪的一个参数
fig, ax = plt.subplots(1, 2, figsize=(16, 3))

col = 'damageDealt'
sub = train_data[['winPlacePerc',col]].copy()
sub[col] = pd.cut(sub[col], 6)
sub.groupby(col).mean()['winPlacePerc'].plot.bar(ax=ax[0])
train_data[col].hist(bins=20, ax=ax[1])

#%% walkDistance, rideDistance, swimDistance
#分别为行走距离，骑车距离，游泳距离
fig, ax = plt.subplots(3, 2, figsize=(16, 10))

cols = ['walkDistance', 'rideDistance', 'swimDistance']
for col, ax in zip(cols, ax):
    sub = train_data[['winPlacePerc',col]].copy()
    sub[col] = pd.cut(sub[col], 6)
    sub.groupby(col).mean()['winPlacePerc'].plot.bar(ax=ax[0])
    train_data[col].hist(bins=20, ax=ax[1])
#开挂
sq = ''
querys = ['walkDistance == 0 & rideDistance == 0 & swimDistance == 0',' & kills > 0',' & headshotKills > 0',' & heals > 0']
for q in querys:
    sq += q
    sub = train_data.query(sq)
    print(sq, '\n count:', len(sub), ' winPlacePerc:', round(sub['winPlacePerc'].mean(),3))
del sub

#%% killPlints, rankPoints, winPoints这三个参数


#%% winPlacePerc
train_data['winPlacePerc'].describe()

#%% 特征工程
all_data = train_data.append(test_data)

#%% rank as percent
#按比赛场次对killPlacePerc和walkDistancePerc进行排序
#match = all_data.groupby('matchId')
#all_data['killPlacePerc'] = match['kills'].rank(pct=True).values
#all_data['walkDistance'] = match['walkDistance'].rank(pct=True).values

#%% 总路程
all_data['totalDistance'] = all_data['rideDistance'] + all_data['walkDistance'] + all_data['swimDistance']

#%%新特征
# 使用药物数
all_data['healthItems'] = all_data['heals'] + all_data['boosts']
# 这个特征有点奇怪
all_data['killDamage'] = all_data['kills']*100+all_data['damageDealt']
# 杀人排名over最高排名
all_data['killPlaceOverMaxPlace'] = all_data['killPlace'] / all_data['maxPlace']
# 杀人数除以行走路程
all_data['killsOverWalkDistance'] = all_data['kills'] / all_data['walkDistance']
# 行走距离除以比赛持续时间
all_data['walkDistancePerSec'] = all_data['walkDistance'] / all_data['matchDuration']

fillInf(all_data, 0)
#%% 丢特征
all_data.drop((['headshotKills', 'teamKills', 'roadKills', 'vehicleDestroys']), axis=1, inplace=True)
all_data.drop((['rideDistance', 'swimDistance', 'matchDuration']), axis=1, inplace=True)
all_data.drop((['rankPoints', 'killPoints', 'winPoints']), axis=1, inplace=True)

#%% grouping
#很重要，要对同一场比赛进行预测
match = all_data.groupby(['matchId'])
group = all_data.groupby(['matchId', 'groupId'])

agg_col = list(all_data.columns)
excluede_agg_col = ['Id', 'matchId', 'groupId', 'matchType', 'maxPlace', 'numGroups', 'winPlacePerc']
for c in excluede_agg_col:
    agg_col.remove(c)
print(agg_col)

sum_col = ['kills', 'killPlace', 'damageDealt', 'walkDistance', 'healthItems']

#%% match sum, group sum
match_data=pd.concat([
        match.size().to_frame('m.players'),
        match[sum_col].sum().rename(columns=lambda s:'m.sum.'+s),
        match[sum_col].max().rename(columns=lambda s:'m.max.'+s)
        ], axis=1).reset_index()
#merge inner join 的方法，使用内链接                
match_data = pd.merge(match_data, group[sum_col].sum().rename(columns=lambda s:'sum.'+s).reset_index())
match_data = reduce_mem_usage(match_data)

print(match_data.shape)

#%% ranking of kills and killPlace in each match
minKills = all_data.sort_values(['matchId', 'groupId', 'kills', 'killPlace']).groupby(['matchId', 'groupId', 'kills']).first().reset_index().copy()
for n in np.arange(5):
    c = 'kills' + str(n) + '_Place'
    minKills[c] = 0
    nKills = (minKills['kills'] == n)
    minKills.loc[nKills, c] = minKills[nKills].groupby(['matchId'])['killPlace'].rank().values
#左链接
    match_data = pd.merge(match_data, minKills[nKills][['matchId','groupId',c]], how='left')
    match_data[c].fillna(0, inplace=True)

match_data = reduce_mem_usage(match_data)
del minKills, nKills
print(match_data.shape)

#%% group mean, max, min
all_data = pd.concat([
#队伍里有多少人
        group.size().to_frame('players'),
        group.mean(),
        group[agg_col].max().rename(columns=lambda s:'max.'+s),
        group[agg_col].min().rename(columns=lambda s:'min.'+s),
        ], axis=1).reset_index()
all_data = reduce_mem_usage(all_data)

print(all_data.shape)
    
#%% aggregate feature
#融合特征
numcols = all_data.select_dtypes(include='number').columns.values
numcols = numcols[numcols != 'winPlacePerc']

#%% match summary, max
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
all_data = pd.merge(all_data, matchRank)
#队伍排名占比赛最高排名的比例
for c in numcols:
    all_data['rank.' + c] = all_data['rank.' + c] / all_data['max.rank.' + c]
    all_data.drop(['max.rank.' + c], axis=1, inplace=True)
del matchRank

print(all_data.shape)

#%% killPlace rank of group and kills
#rank.minor.maxKillPlace 和 rank.minor.minKillPlace 
#这两个值越大，越有可能吃鸡
killMinorRank = all_data[['matchId','min.kills','max.killPlace']].copy()
group = killMinorRank.groupby(['matchId','min.kills'])
killMinorRank['rank.minor.maxKillPlace'] = group.rank(pct=True).values
all_data = pd.merge(all_data, killMinorRank)

killMinorRank = all_data[['matchId','max.kills','min.killPlace']].copy()
group = killMinorRank.groupby(['matchId','max.kills'])
killMinorRank['rank.minor.minKillPlace'] = group.rank(pct=True).values
all_data = pd.merge(all_data, killMinorRank)

del killMinorRank


#%% delete feature
# 删除只有一个属性的项
# 能否删除属性比例不平衡的项
constant_column = [col for col in all_data.columns if all_data[col].nunique() == 1]
print('drop columns:', constant_column)
all_data.drop(constant_column, axis=1, inplace=True)

#%% match type
mapper = lambda x: 'solo' if ('solo' in x) else 'duo' if ('duo' in x) or ('crash' in x) else 'squad'
all_data['matchTypeCat'] = all_data['matchType'].map(mapper)

print(all_data['matchTypeCat'].values_counts())

#%% 查看缺失值
null_cnt = all_data.isnull().sum().sort_values()
print(null_cnt[null_cnt > 0])


#%% 属性特征编码？
cols = [col for col in all_data.columns if col not in ['Id', 'matchId', 'groupId']]


for i,t in all_data.loc[:,cols].dtypes.iteritems():
    if t == object:
        all_data[i] = pd.factorize(all_data[i])[0]


all_data = reduce_mem_usage(all_data)
all_data.head()


#%%
X_train = all_data[all_data['winPlacePerc'].notnull()].reset_index(drop=True)
X_test = all_data[all_data['winPlacePerc'].isnull()].drop(['winPlacePerc'], axis=1).reset_index(drop=True)
del all_data
gc.collect()

Y_train = X_train.pop('winPlacePerc')
X_test_grp = X_test[['matchId','groupId']].copy()
#train_matchId = X_train['matchId']

# drop matchId,groupId
X_train.drop(['matchId','groupId'], axis=1, inplace=True)
X_test.drop(['matchId','groupId'], axis=1, inplace=True)


print(X_train.shape, X_test.shape)


#%%
#dir()是什么鬼
print(pd.DataFrame([[val for val in dir()], [sys.getsizeof(eval(val)) for val in dir()]],
                   index=['name','size']).T.sort_values('size', ascending=False).reset_index(drop=True)[:10])

#%%引入xgboost,按matchType进行预测
from sklearn.model_selection import GroupKFold
from sklearn.preprocessing import minmax_scale
import lightgbm as lgb

params={'learning_rate': 0.05,
        'objective':'mae',
        'metric':'mae',
        'num_leaves': 31,
        'verbose': 0,
        'random_state':42,
        'bagging_fraction': 0.7,
        'feature_fraction': 0.7
       }
mts = list()
fis = list()
pred = np.zeros(X_test.shape[0])
for mt in X_train['matchTypeCat'].unique():
    idx = X_train[X_train['matchTypeCat'] == mt].index
    reg = lgb.LGBMRegressor(**params, n_estimators=3000)
    reg.fit(X_train.loc[idx], Y_train.loc[idx])

    idx = X_test[X_test['matchTypeCat'] == mt].index
    pred[idx] = reg.predict(X_test.loc[idx], num_iteration=reg.best_iteration_)
    mts.append(mt)
    fis.append(reg.feature_importances_)


    
    
    
    