# -*- coding: utf-8 -*-
"""
Created on Sun Jan 27 22:52:09 2019

@author: ZHOU-JC
"""

# -*- coding: utf-8 -*-
"""
Created on Fri Nov 30 16:19:26 2018

@author: ZHOU-JC
"""
import time
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
train_data  = pd.read_csv(r'C:\Users\ZHOU-JC\Desktop\data match\PUBG Finish Placement Prediction(Kernels Only)\train_V2.csv', nrows=1000000)
test_data   = pd.read_csv(r'C:\Users\ZHOU-JC\Desktop\data match\PUBG Finish Placement Prediction(Kernels Only)\test_V2.csv', nrows=1000)
# 删掉缺失值
train_data.dropna(inplace=True)
# 切分
train_data,test_data = train_test_split(train_data, test_size=0.2, random_state=1)
train_data = reduce_mem_usage(train_data)
test_data = reduce_mem_usage(test_data)
# 保存真值
real = test_data[['Id','matchId','groupId','winPlacePerc']]
#%%
match_train = list(train_data['matchId'].unique())
match_test = list(test_data['matchId'].unique())
labels = []
for each in match_test:
    if each in match_train:
        labels.append(each)




test_data.drop(['winPlacePerc'], axis=1, inplace=True)
# 真实的train.csv和test.csv应该是没有有同一场比赛的吧
##
#bb = test_data.loc[test_data['winPlacePerc'].isnull(), ['groupId','matchId']].groupby('groupId').size().reset_index()
#bb = pd.merge(pd.DataFrame(bb),aa, how='left')
# all_data

all_data = train_data.append(test_data, sort=False).reset_index(drop=True)

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

#%% 删除不重要的
all_data.drop((['headshotKills', 'teamKills', 'roadKills', 'vehicleDestroys']), axis=1, inplace=True)
all_data.drop((['rideDistance', 'swimDistance', 'matchDuration']), axis=1, inplace=True)
all_data.drop((['rankPoints', 'killPoints', 'winPoints']), axis=1, inplace=True)


#%% 按match和group分组预测
match = all_data.groupby('matchId')
group = all_data.groupby(['matchId','groupId','matchType'])
#['assists','boosts','damageDealt','DBNOs','heals','killPlace','kills','killStreaks',
# 'longestKill', 'revives', 'walkDistance', 'weaponsAcquired', 'Total_distance', 'engage', 'healthItems']
# target目标特征
agg_col = list(all_data.columns)
exclude_agg_col = ['Id','matchId','groupId','matchType','maxPlace','numGroups','winPlacePerc']
for c in exclude_agg_col:
    agg_col.remove(c)
#%% 
sum_col = ['kills', 'killPlace', 'damageDealt', 'walkDistance', 'healthItems']


#%% 
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

#%% group mean, max, min
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

#%% 融合特征
numcols = list(all_data.select_dtypes(include='number').columns)
numcols.remove('winPlacePerc')
#['players', 'assists', 'boosts', 'damageDealt', 'DBNOs', 'heals', 'killPlace',
# 'kills', 'killStreaks', 'longestKill', 'maxPlace', 'numGroups', 'revives', 'walkDistance',
# 'weaponsAcquired', 'Total_distance', 'engage','healthItems', 'sum.assists', 'sum.boosts', 'sum.damageDealt',
# 'sum.DBNOs',# 'sum.heals',# 'sum.killPlace',# 'sum.kills', 'sum.killStreaks', 'sum.longestKill',
# 'sum.revives', 'sum.walkDistance', 'sum.weaponsAcquired', 'sum.Total_distance', 'sum.engage',
# 'sum.healthItems', 'max.assists', 'max.boosts', 'max.damageDealt', 'max.DBNOs', 'max.heals',
# 'max.killPlace', 'max.kills', 'max.killStreaks', 'max.longestKill', 'max.revives', 'max.walkDistance',
# 'max.weaponsAcquired', 'max.Total_distance', 'max.engage', 'max.healthItems', 'min.assists', 'min.boosts',
# 'min.damageDealt','min.DBNOs','min.heals', 'min.killPlace', 'min.kills', 'min.killStreaks',
# 'min.longestKill', 'min.revives', 'min.walkDistance', 'min.weaponsAcquired', 'min.Total_distance',
# 'min.engage', 'min.healthItems']

#%% all_data是按组来分的
# match_data包含每场比赛的总数和最大值
all_data = pd.merge(all_data, match_data)
del match_data

all_data['enemy.players'] = all_data['m.players'] - all_data['players']
for c in sum_col:

#敌人平均的kills,killPlace,damageDealt,walkDistance,healthItemns
    all_data['enemy.' + c] = (all_data['m.sum.' + c] - all_data['sum.' + c]) / all_data['enemy.players']
    #all_data['p.sum_msum.' + c] = all_data['sum.' + c] / all_data['m.sum.' + c]
#团队平均值与敌人平均值的比例
#团队总值占比赛总值的多少
    all_data['p.sum_msum.' + c] = all_data['sum.' + c] / all_data['m.sum.' + c]
#团队最大值占比赛最大值的多少
    all_data['p.max_mmax.' + c] = all_data['max.' + c] / all_data['m.max.' + c]
#丢掉比赛总值和比赛的总值，最大值，最小值，因为这对预测没有帮助
    all_data.drop(['m.sum.' + c, 'm.max.' + c], axis=1, inplace=True)
fillInf(all_data, 0)    
print(all_data.shape)

#%% match rank 将一些个人、队伍数值属性转换为白比分排名
aa = all_data.copy()
match = all_data.groupby('matchId')
# numcols 包含有group的平均值，总值，最大值，最小值的信息
matchRank = match[numcols].rank(pct=True).rename(columns=lambda s:'rank.'+s)
all_data = reduce_mem_usage(pd.concat([all_data, matchRank], axis=1))
rank_col = matchRank.columns
del matchRank
#rank_col
#istance',
#       'rank.weaponsAcquired', 'rank.Total_distance', 'rank.engage',
#       'rank.healthItems', 'rank.sum.assists', 'rank.sum.boosts',
#       'rank.sum.damageDealt', 'rank.sum.DBNOs', 'rank.sum.heals',
#       'rank.sum.killPlace', 'rank.sum.kills', 'rank.sum.killStreaks',
#       'rank.sum.longestKill', 'rank.sum.revives', 'rank.sum.walkDistance',
#       'rank.sum.weaponsAcquired', 'rank.sum.Total_distance',
#       'rank.sum.engage', 'rank.sum.healthItems', 'rank.max.assists',
#       'rank.max.boosts', 'rank.max.damageDealt', 'rank.max.DBNOs',
#       'rank.max.heals', 'rank.max.killPlace', 'rank.max.kills',
#       'rank.max.killStreaks', 'rank.max.longestKill', 'rank.max.revives',
#       'rank.max.walkDistance', 'rank.max.weaponsAcquired',
#       'rank.max.Total_distance', 'rank.max.engage', 'rank.max.healthItems',
#       'rank.min.assists', 'rank.min.boosts', 'rank.min.damageDealt',
#       'rank.min.DBNOs', 'rank.min.heals', 'rank.min.killPlace',
#       'rank.min.kills', 'rank.min.killStreaks', 'rank.min.longestKill',
#       'rank.min.revives', 'rank.min.walkDistance', 'rank.min.weaponsAcquired',
#       'rank.min.Total_distance', 'rank.min.engage', 'rank.min.healthItems'
# instead of rank(pct=True, method='dense')
match = all_data.groupby('matchId')
matchRank = match[rank_col].max().rename(columns=lambda s: 'max.' + s).reset_index()
all_data = pd.merge(all_data, matchRank) #这步all_data是DataFrame形式
all_data = reduce_mem_usage(all_data)
#个人、队伍排名占比赛最高排名的比例
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
# 只有matchType 和 matchTypeCat 是 object类别型
for i,t in all_data.loc[:,cols].dtypes.iteritems():
    if t == object:
        all_data[i] = pd.factorize(all_data[i])[0]

all_data = reduce_mem_usage(all_data)
all_data.head()
# 只对了matchType和matchTypeCat 编码 

#%% 组装X_train,Y_train,X_test
X_train = all_data[all_data['winPlacePerc'].notnull()].reset_index(drop=True)
X_test = all_data[all_data['winPlacePerc'].isnull()].drop(['winPlacePerc'], axis=1).reset_index(drop=True)
Y_train = X_train.pop('winPlacePerc')
X_test_grp = X_test[['matchId','groupId']].copy()
#train_matchId = X_train['matchId']

# drop matchId,groupId
X_train.drop(['matchId','groupId'], axis=1, inplace=True)
X_test.drop(['matchId','groupId'], axis=1, inplace=True)

print(X_train.shape, X_test.shape)

#%% lgb
start = time.clock()
import xgboost as xgb
import lightgbm as lgb

params={'learning_rate': 0.5,    #学习速率
        'n_estimators': 2000, 
        'num_leaves': 50, 
        'max_depth': 5,          #叶的最大深度
        'min_data_in_leaf': 8,  
        'metric':'mae',          #目标函数
        'objective': 'regression', 
        'verbose': 0,
        'bagging_fraction': 0.25,    #每次迭代用的数据比例
        'feature_fraction': 0.25,     #每次迭代用的特征比例
        'reg_alpha': 2.5,
        'reg_lambda': 0.01,
        'seed': 8}
        

mts = []
fis = []

# 
#pred = np.zeros(X_test.shape[0])
#for mt in X_train['matchTypeCat'].unique():
#    idx = X_train[X_train['matchTypeCat'] == mt].index
#    model_lgb = lgb.LGBMRegressor(**params)
#    model_lgb.fit(X_train.loc[idx], Y_train.loc[idx])
#    
#    idx = X_test[X_test['matchTypeCat'] == mt].index
#    pred[idx] = model_lgb.predict(X_test.loc[idx])
#    mts.append(mt)
#    fis.append(pd.DataFrame(model_lgb.feature_importances_, index=X_train.columns).sort_values(0, ascending=False))
#a1 = fis[0]
#a2 = fis[1]
#a3 = fis[2]    

pred = np.zeros(X_test.shape[0])
model_lgb = lgb.LGBMRegressor(**params)
model_lgb.fit(X_train, Y_train)
pred = model_lgb.predict(X_test)
fis.append(pd.DataFrame(model_lgb.feature_importances_, index=X_train.columns).sort_values(0, ascending=False))
feature_importance = fis[0]    

result = pd.concat([pd.DataFrame(pred), X_test_grp], axis=1)
a = pd.merge(result, real, on='groupId', how='right')
print('测试集得到为',abs(a[0].values-a['winPlacePerc'].values).mean())

elapsed = (time.clock()) - start
print('Time used:', elapsed)

##%% lasso
#from sklearn.linear_model import Lasso
#from sklearn.linear_model import LassoCV
#from sklearn.preprocessing import RobustScaler
#start = time.clock()
#
##
##al = np.linspace(0.1, 1.5, 20)
##model = RidgeCV(alphas = al)
##model.fit(X_train, Y_train)
##print('最佳的正则化参数为',model.alpha_)
#
#pred = np.zeros(X_test.shape[0])
#best_alpha = 1.131578947368421
#lasso = make_pipeline(RobustScaler(), Lasso(alpha = best_alpha, max_iter=50000))
#lasso.fit(X_train, Y_train)
#pred = lasso.predict(X_test)
#
#
#result = pd.concat([pd.DataFrame(pred), X_test_grp], axis=1)
#a = pd.merge(result, bb, on='groupId')
#print('测试集得到为',abs(a['0_x'].values-a['winPlacePerc'].values).mean())
#elapsed = (time.clock()) - start
#print('Time used:', elapsed)
#
##%% 后处理


