# -*- coding: utf-8 -*-
#%% 
import sys as sys
import time
import gc
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
from sklearn.pipeline import make_pipeline
import seaborn as sns
#%%
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
start = time.clock()
train_data  = pd.read_csv(r'C:\Users\ZHOU-JC\Desktop\data match\PUBG Finish Placement Prediction(Kernels Only)\train_V2.csv',
                          nrows=500000)

mapper = lambda x: 'solo' if ('solo' in x) else 'duo' if ('duo' in x) or ('crash' in x) else 'squad'
train_data['matchTypeCat'] = train_data['matchType'].map(mapper)

#
idx_solo = train_data[train_data['matchTypeCat']=='solo'].index
idx_duo = train_data[train_data['matchTypeCat']=='duo'].index
idx_squad = train_data[train_data['matchTypeCat']=='squad'].index

# 分组 
solo = train_data.iloc[idx_solo]
duo = train_data.iloc[idx_duo]
squad = train_data.iloc[idx_squad]

#
a1 = solo.groupby('matchId').first()
a2 = duo.groupby('matchId').first()
a3 = squad.groupby('matchId').first()

#%% 切分
from sklearn.model_selection import train_test_split

X_train_solo, X_test_solo = train_test_split(a1, test_size=0.2, random_state=1)
X_train_duo,  X_test_duo  = train_test_split(a2, test_size=0.2, random_state=2)
X_train_squad,  X_test_squad  = train_test_split(a3, test_size=0.2, random_state=3)

X_train_solo_idx = list(X_train_solo.index)
X_test_solo_idx  = list(X_test_solo.index)
X_train_duo_idx = list(X_train_duo.index)
X_test_duo_idx = list(X_test_duo.index)
X_train_squad_idx = list(X_train_squad.index)
X_test_squad_idx = list(X_test_squad.index)

X_train_solo = train_data[train_data['matchId'].isin(X_train_solo_idx)]
X_test_solo = train_data[train_data['matchId'].isin(X_test_solo_idx)]
X_train_duo = train_data[train_data['matchId'].isin(X_train_duo_idx)]
X_test_duo = train_data[train_data['matchId'].isin(X_test_duo_idx)]
X_train_squad = train_data[train_data['matchId'].isin(X_train_squad_idx)]
X_test_squad = train_data[train_data['matchId'].isin(X_test_squad_idx)]

# 组成train_data 和 test_data
train_data = pd.concat([X_train_solo, X_train_duo, X_train_squad], axis=0).reset_index(drop=True)
test_data = pd.concat([X_test_solo, X_test_duo, X_test_squad], axis=0).reset_index(drop=True)
#
real = test_data[['Id','winPlacePerc']]
#
train_data = train_data.drop('matchTypeCat', axis=1)
test_data = test_data.drop(['matchTypeCat', 'winPlacePerc'], axis=1)

#%%
start = time.clock()
#
train_data = reduce_mem_usage(train_data)
test_data = reduce_mem_usage(test_data)
# 删掉缺失值
train_data.dropna(inplace=True)

# all_data
all_data = train_data.append(test_data, sort=False).reset_index(drop=True)
print(train_data.shape)
print(test_data.shape)
elapsed = (time.clock() - start)
print("导入数据所谓时间为:",elapsed)

#%%
match = all_data.groupby('matchId')
all_data['killPlacePerc'] = match['kills'].rank(pct=True).values
all_data['walkDistancePerc'] = match['walkDistance'].rank(pct=True).values

#%%
start = time.clock()

#%% rank as percent
# 个人在比赛中的排名
match = all_data.groupby('matchId')
# 杀人数在比赛中的排名
#all_data['killPlacePerc'] = match['kills'].rank(pct=True).values
# 行走数在比赛中的排名
#all_data['walkDistancePerc'] = match['walkDistance'].rank(pct=True).values

#%% 增加属性 
# 增加属性：总距离
all_data['Total_distance'] = all_data['walkDistance']+all_data['swimDistance'] +all_data['rideDistance']
# 增加属性：engage(助攻+自己击倒)
all_data['engage'] = all_data['kills'] + all_data['assists'] 
# 增加属性：总用药数
all_data['healthItems'] = all_data['heals'] + all_data['boosts']
all_data['killStreakrate'] = all_data['killStreaks']/all_data['kills']
all_data['headshotKills_over_kills'] = all_data['headshotKills'] / all_data['kills']
all_data['killPlace_over_maxPlace'] = all_data['killPlace'] / all_data['maxPlace']
all_data['walkDistance_over_kills'] = all_data['walkDistance'] / all_data['kills']
all_data["skill"] = all_data["headshotKills"] + all_data["roadKills"]

all_data[all_data == np.Inf] = np.NaN
all_data[all_data == np.NINF] = np.NaN
    
print("Removing Na's From DF")
cols = []
for each in all_data.columns:
    cols.append(each)
cols.remove('winPlacePerc')    

a = all_data['winPlacePerc'].copy()
all_data.fillna(0, inplace=True)
all_data['winPlacePerc'] = a
#%% 删除不重要的
all_data.drop((['headshotKills', 'teamKills', 'roadKills', 'vehicleDestroys']), axis=1, inplace=True)
all_data.drop((['swimDistance']), axis=1, inplace=True)
all_data.drop((['matchDuration']), axis=1, inplace=True)

#all_data.drop((['rankPoints', 'killPoints', 'winPoints']), axis=1, inplace=True)
elapsed = (time.clock() - start)
print("增加特征所花时间:",elapsed)

#%%
start = time.clock()

# 按match和group分组预测
match = all_data.groupby('matchId')
group = all_data.groupby(['matchId','groupId','matchType'])
#['assists','boosts','damageDealt','DBNOs','heals','killPlace','kills','killStreaks',
# 'longestKill', 'revives', 'walkDistance', 'weaponsAcquired', 'killPlacePerc',
# 'walkDistancePerc', 'distance', 'engage', 'healthItems']
# target目标特征
agg_col = list(all_data.columns)
exclude_agg_col = ['Id','matchId','groupId','matchType','maxPlace','numGroups','winPlacePerc']
for c in exclude_agg_col:
    agg_col.remove(c)

sum_col = ['kills', 'killPlace', 'damageDealt', 'walkDistance', 'healthItems']

# 
match_data=pd.concat([
        # 每场比赛的人数
        match.size().to_frame('m.players'),
        # 每场比赛的总和
        match[sum_col].sum().rename(columns=lambda s:'m.sum.'+s),
        # 每场比赛的最大值
        match[sum_col].max().rename(columns=lambda s:'m.max.'+s)
        ], axis=1).reset_index() #reset_inde 这波操作会令index变成顺序数字，而columns会多出个matchId
#merge inner join 的方法，使用内链接 
match_data = pd.merge(match_data, group[sum_col].sum().rename(columns=lambda s:'sum.'+s).reset_index())
match_data = reduce_mem_usage(match_data)
print(match_data.shape) #列数为组数

# group mean, max, min
# 假如是单人的话mean, max, min一样的
#all_data 是按group来分的
all_data = pd.concat([
        # 每支队伍的人数
        group.size().to_frame('players'),
        # 每组的平均值
        group.mean(), 
        # 每组的总值
        group[agg_col].sum().rename(columns=lambda s:'sum.'+s),
        # 每组的最大值
        group[agg_col].max().rename(columns=lambda s:'max.'+s),
        # 每组的最小值
        group[agg_col].min().rename(columns=lambda s:'min.'+s),
        ], axis=1).reset_index()

all_data = reduce_mem_usage(all_data)
print(all_data.shape)

#记录numcols 刚生成group后
numcols = list(all_data.select_dtypes(include='number').columns)
numcols.remove('winPlacePerc')

# all_data是按组来分的
# match_data包含每场比赛的总数和最大值
all_data = pd.merge(all_data, match_data)
del match_data
gc.collect()

all_data['enemy.players'] = all_data['m.players'] - all_data['players']
for c in sum_col:

#敌人平均的kills,killPlace,damageDealt,walkDistance,healthItemns
    all_data['enemy.' + c] = (all_data['m.sum.' + c] - all_data['sum.' + c]) / all_data['enemy.players']
    #all_data['p.sum_msum.' + c] = all_data['sum.' + c] / all_data['m.sum.' + c]
#团队平均值与敌人平均值的比例
#团队总值占比赛总值的多少
#    all_data['p.sum_msum.' + c] = all_data['sum.' + c] / all_data['m.sum.' + c]
#团队最大值占比赛总值的多少
#    all_data['p.max_msum.' + c] = all_data['max.' + c] / all_data['m.sum.' + c]

#团队最大值占比赛最大值的多少
#    all_data['p.max_mmax.' + c] = all_data['max.' + c] / all_data['m.max.' + c]
#丢掉比赛总值和比赛的总值，最大值，最小值，因为这对预测没有帮助
    all_data.drop(['m.sum.' + c, 'm.max.' + c], axis=1, inplace=True)
fillInf(all_data, 0)    

all_data = reduce_mem_usage(all_data)
print(all_data.shape)

elapsed = (time.clock() - start)
print("按group、match分组算花的时间",elapsed)



#%% match rank 将一些个人、队伍数值属性转换为白比分排名
start = time.clock()
match = all_data.groupby('matchId')
# numcols 包含有group的平均值，总值，最大值，最小值的信息
matchRank = match[numcols].rank(pct=True).rename(columns=lambda s:'rank.'+s)
all_data = reduce_mem_usage(pd.concat([all_data, matchRank], axis=1))
rank_col = matchRank.columns
del matchRank
gc.collect()

# instead of rank(pct=True, method='dense')
match = all_data.groupby('matchId')
matchRank = match[rank_col].max().rename(columns=lambda s: 'max.' + s).reset_index()
all_data = pd.merge(all_data, matchRank) #这步all_data是DataFrame形式
all_data = reduce_mem_usage(all_data)
del matchRank
gc.collect()

print('all_data的size为',all_data.shape)
elapsed = (time.clock() - start)
print(" match rank 将一些个人、队伍数值属性转换为百比分排名",elapsed)

#%% 个人、队伍排名占比赛最高排名的比例
start = time.clock()

for c in numcols:
    all_data['rank.' + c] = all_data['rank.' + c] / all_data['max.rank.' + c]
    all_data.drop(['max.rank.' + c], axis=1, inplace=True)
all_data = reduce_mem_usage(all_data)
gc.collect()

    
print(all_data.shape) 

# 经检查 这里并没有nan值

elapsed = (time.clock() - start)
print("队伍排名占比赛最高排名的值",elapsed)


#%%
#''' TODO: incomplete
#''' 
killMinorRank = all_data[['matchId','min.kills','max.killPlace']].copy()
group = killMinorRank.groupby(['matchId','min.kills'])
killMinorRank['rank.minor.maxKillPlace'] = group.rank(pct=True).values
all_data = pd.merge(all_data, killMinorRank)

killMinorRank = all_data[['matchId','max.kills','min.killPlace']].copy()
group = killMinorRank.groupby(['matchId','max.kills'])
killMinorRank['rank.minor.minKillPlace'] = group.rank(pct=True).values
all_data = pd.merge(all_data, killMinorRank)

del killMinorRank
gc.collect()

#%% killPlace rank of group and kills
#%% delete feature
# 删除只有一个属性的项
# 能否删除属性比例不平衡的项
constant_column = [col for col in all_data.columns if all_data[col].nunique() == 1]
print('drop columns:', constant_column)
all_data.drop(constant_column, axis=1, inplace=True)

# match type
mapper = lambda x: 'solo' if ('solo' in x) else 'duo' if ('duo' in x) or ('crash' in x) else 'squad'
all_data['matchTypeCat'] = all_data['matchType'].map(mapper)

print(all_data['matchTypeCat'].value_counts())


# encode 编码
cols = [col for col in all_data.columns if col not in ['Id', 'matchId', 'groupId']]
# 只有matchType 和 matchTypeCat 是 object类别型
for i,t in all_data.loc[:,cols].dtypes.iteritems():
    if t == object:
        all_data[i] = pd.factorize(all_data[i])[0]

all_data = reduce_mem_usage(all_data)
# 只对了matchType和matchTypeCat 编码

#%% 组装X_train,Y_train,X_test
X_train = all_data[all_data['winPlacePerc'].notnull()].reset_index(drop=True)
X_test = all_data[all_data['winPlacePerc'].isnull()].drop(['winPlacePerc'], axis=1).reset_index(drop=True)
del all_data
gc.collect()


Y_train = X_train.pop('winPlacePerc')
X_test_Id = X_test[['matchId','groupId']].copy()
#train_matchId = X_train['matchId']

# drop matchId,groupId
X_train.drop(['matchId','groupId'], axis=1, inplace=True)
X_test.drop(['matchId','groupId'], axis=1, inplace=True)

print('X_train和X_test的容量',X_train.shape, X_test.shape)

#%%
#%% lgb
start = time.clock()
print('开始训练')

import xgboost as xgb
import lightgbm as lgb

params={'learning_rate': 0.05,    #学习速率
        'n_estimators': 500, 
        'num_leaves': 30, 
        'max_depth': 5,          #叶的最大深度
        'min_data_in_leaf': 500,  
        'metric':'mae',          #目标函数
        'objective': 'mae', 
        'verbose': 0,
        'bagging_fraction': 0.9,    #每次迭代用的数据比例
        'feature_fraction': 0.6,     #每次迭代用的特征比例
        'seed': 16}
        

mts = []
fis = []

cols_rankPoints = ['rankPoints','sum.rankPoints','max.rankPoints','min.rankPoints',
        'rank.rankPoints','rank.sum.rankPoints','rank.max.rankPoints','rank.min.rankPoints']
cols_killwinPoints = ['winPoints','sum.winPoints','max.winPoints','min.winPoints','rank.winPoints',
                      'rank.sum.winPoints','rank.max.winPoints','rank.min.winPoints',
                      'killPoints','sum.killPoints','max.killPoints','min.killPoints',
                    'rank.killPoints','rank.sum.killPoints','rank.max.killPoints','rank.min.killPoints']
# 
pred = np.zeros(X_test.shape[0])
for mt in X_train['matchTypeCat'].unique():
    idx = X_train[X_train['matchTypeCat'] == mt].index
    model_lgb = lgb.LGBMRegressor(**params)
    model_lgb.fit(X_train.loc[idx], Y_train.loc[idx])
    idx = X_test[X_test['matchTypeCat'] == mt].index
    pred[idx] = model_lgb.predict(X_test.loc[idx])    
    
    
 
    
    idx = X_train[(X_train['matchTypeCat'] == mt) & (X_train['rankPoints'] == (-1 or 0))].index
    model_lgb = lgb.LGBMRegressor(**params)
    model_lgb.fit(X_train.loc[idx].drop(cols_rankPoints, axis=1), Y_train.loc[idx])
    idx = X_test[X_test['matchTypeCat'] == mt].index
    pred[idx] = model_lgb.predict(X_test.loc[idx].drop(cols_rankPoints, axis=1))
    
    
    idx = X_train[(X_train['matchTypeCat'] == mt) & (X_train['rankPoints'] != (-1 and 0))].index
    model_lgb = lgb.LGBMRegressor(**params)
    model_lgb.fit(X_train.loc[idx].drop(cols_killwinPoints, axis=1), Y_train.loc[idx])
    idx = X_test[X_test['matchTypeCat'] == mt].index
    pred[idx] = model_lgb.predict(X_test.loc[idx].drop(cols_killwinPoints, axis=1))
    
    mts.append(mt)
    fis.append(pd.DataFrame(model_lgb.feature_importances_, index=X_train.columns).sort_values(0, ascending=False))
a1 = fis[0]
a2 = fis[1]
a3 = fis[2]    



# 调参
#for mt in X_train['matchTypeCat'].unique():
#    idx = X_train[X_train['matchTypeCat'] == mt].index    
#    parameters = {
#              'num_leaves':  [40,80,128],
#              'max_depth': [5,6,7],
#              }
#    reg = lgb.LGBMRegressor(**params)
#    gsearch = GridSearchCV(reg, param_grid=parameters, scoring='neg_mean_absolute_error', 
#                           cv=3, n_jobs=-1)
#    gsearch.fit(X_train.loc[idx], Y_train.loc[idx])
#    gsearch.best_params_
#    print('%s最佳参数为%s'%(mt, gsearch.best_params_))
#    mts.append(mt)
    
#pred = np.zeros(X_test.shape[0])
#model_lgb = lgb.LGBMRegressor(**params)
#model_lgb.fit(X_train, Y_train)
#pred = model_lgb.predict(X_test)
#fis.append(pd.DataFrame(model_lgb.feature_importances_, index=X_train.columns).sort_values(0, ascending=False))
#feature_importance = fis[0]    

result = pd.concat([pd.DataFrame(pred), X_test_Id], axis=1)
result = pd.merge(test_data[['Id','groupId']], result, how='left', on='groupId')
result.drop(['groupId','matchId'], axis=1, inplace=True)
result.columns=['Id', 'winPlacePerc']
result1 = result.copy()
result1.loc[result.query('winPlacePerc > 1').index, 'winPlacePerc'] =1
result1.loc[result.query('winPlacePerc < 0').index, 'winPlacePerc'] =0

s = pd.merge(real, result1, how='left', on = 'Id')
print('第一个得分为',abs(s['winPlacePerc_x']-s['winPlacePerc_y']).mean())

elapsed = (time.clock()) - start
print('Time used:', elapsed)
elapsed = (time.clock() - start)
print("训练所用时间",elapsed)

#%%
result = result.merge(test_data[["Id", "matchId", "groupId", "maxPlace", "numGroups"]], on="Id", how="left")

# Sort, rank, and assign adjusted ratio
result_group = result.groupby(["matchId", "groupId"]).first().reset_index()
result_group["rank"] = result_group.groupby(["matchId"])["winPlacePerc"].rank()
result_group = result_group.merge(
    result_group.groupby("matchId")["rank"].max().to_frame("max_rank").reset_index(), 
    on="matchId", how="left")
result_group["adjusted_perc"] = (result_group["rank"] - 1) / (result_group["numGroups"] - 1)

result = result.merge(result_group[["adjusted_perc", "matchId", "groupId"]], on=["matchId", "groupId"], how="left")
result["winPlacePerc"] = result["adjusted_perc"]

# Deal with edge cases
result.loc[result.maxPlace == 0, "winPlacePerc"] = 0
result.loc[result.maxPlace == 1, "winPlacePerc"] = 1

# Align with maxPlace
subset = result.loc[result.maxPlace > 1]
gap = 1.0 / (subset.maxPlace.values - 1)
new_perc = np.around(subset.winPlacePerc.values / gap) * gap
result.loc[result.maxPlace > 1, "winPlacePerc"] = new_perc

# Edge case
result.loc[(result.maxPlace > 1) & (result.numGroups == 1), "winPlacePerc"] = 0
assert result["winPlacePerc"].isnull().sum() == 0
result["winPlacePerc"] = result["winPlacePerc"]
result = result[["Id", "winPlacePerc"]]
result = pd.merge(result, real, how='left', on='Id')