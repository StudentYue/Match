# -*- coding: utf-8 -*-
"""
Created on Tue Feb 19 15:38:57 2019

@author: ZHOU-JC
"""

# -*- coding: utf-8 -*-
"""
Created on Mon Feb 18 16:44:25 2019

@author: ZHOU-JC
"""

#%% 
import gc
import glob
import os
import json
import matplotlib.pyplot as plt
import pprint

import numpy as np
import pandas as pd

from joblib import Parallel, delayed
from tqdm import tqdm
from PIL import Image
from sklearn.metrics import cohen_kappa_score, make_scorer
from sklearn.model_selection import GridSearchCV, cross_val_score, train_test_split

#%%
import pickle 
with open('x_train', 'wb') as f:
    pickle.dump(x_train, f)
#f = open('x_test' , 'wb')
#pickle(x_test,  f)

#%% 评价函数 Metric used for this competition 
# (Quadratic Weigthed Kappa aka Quadratic Cohen Kappa Score)
def metric(y1,y2):
    return cohen_kappa_score(y1, y2, weights = 'quadratic')

# Make scorer for scikit-learn
scorer = make_scorer(metric)
#%%
df_train  = pd.read_csv('train.csv')
df_test   = pd.read_csv('test.csv')

train = df_train.copy()
test  = df_test.copy()

labels_breed = pd.read_csv('breed_labels.csv')
labels_state = pd.read_csv('color_labels.csv')
labels_color = pd.read_csv('state_labels.csv')

#%% 删除异常值
cul_drop = ['375905770', 'da8d4a273', '27e74e45c', '7b5bee232', '0327b8e94']
df_train = df_train[~df_train['PetID'].isin(cul_drop)]

#%% 提取 sentiment 的特征
# train
def extract_sentiment_feature(i, x):    
    feature_sentiment = pd.DataFrame(columns=['PetID', 'token', 'sentence_magnitude', 'sentence_score','document_magnitude', 'document_score'])

    if x == 'train':
        set_file = 'train'
    else:
        set_file = 'test'
    
        
    file_name = '{}_sentiment/{}.json'.format(set_file,i)
    try:
        f = open(file_name, 'r')
        sentiment_file = json.load(f)
            
        token = [x['name'] for x in sentiment_file['entities']]
        token = ' '.join(token)
            
        sentences_sentiment = [x['sentiment'] for x in sentiment_file['sentences']]
        sentences_sentiment = pd.DataFrame.from_dict(
            sentences_sentiment, orient='columns').sum()
        sentenceSentiment_magnitude = sentences_sentiment['magnitude']
        sentenceSentiment_score     = sentences_sentiment['score']
            
        docementSentiment_magnitude = sentiment_file['documentSentiment']['magnitude']
        documentSentiment_score     = sentiment_file['documentSentiment']['score']
            
        new = pd.DataFrame(
                {'PetID':[i], 
                 'token'               : [token],
                 'sentence_magnitude'  : [sentenceSentiment_magnitude],
                 'sentence_score'      : [sentenceSentiment_score],
                 'document_magnitude'  : [docementSentiment_magnitude], 
                 'document_score'      : [documentSentiment_score]})  
        feature_sentiment = feature_sentiment.append(new)
    except:
        print('{}没找到'.format(file_name))
    return feature_sentiment

#%%
train_feature_sentiment = Parallel(n_jobs=8, verbose=1)(
        delayed(extract_sentiment_feature)(i, 'train') for i in train.PetID)
train_feature_sentiment = [x for x in train_feature_sentiment]
train_feature_sentiment = pd.concat(train_feature_sentiment, ignore_index=True, sort=False)

test_feature_sentiment = Parallel(n_jobs=8, verbose=1)(
        delayed(extract_sentiment_feature)(i, 'test') for i in test.PetID)
test_feature_sentiment = [x for x in test_feature_sentiment]
test_feature_sentiment = pd.concat(test_feature_sentiment, ignore_index=True, sort=False)
#%% 提取 metadata 的特征
#file_name = 'train_metadata/000a290e4-1.json'
#f = open(file_name, 'r')
#metadatafile = json.load(f)
def extract_metadata_feature(i, x):
    feature_metadata = pd.DataFrame()
    if x == 'train':
        set_file = 'train'
    else:
        set_file = 'test'
        
        
    metadata_filenames = sorted(glob.glob('{}_metadata/{}*.json'.format(set_file, i)))
    if len(metadata_filenames) > 0:
        feature_metadata_sub = pd.DataFrame(columns=['PetID', 'annots_score', 'color_score', 'color_pixelfrac', 'crop_conf','crop_importance', 'annots_top_desc'])
        for ff in metadata_filenames:
            f = open(ff, 'rb')
            file = json.load(f)
            #label
            if 'labelAnnotations' in file:
                file_annots = file['labelAnnotations'][:int(len(file['labelAnnotations']) * 0.3)]
                file_top_score = np.asarray([x['score'] for x in file_annots]).mean()
                file_top_desc = [x['description'] for x in file_annots]            
            else:
                file_top_score = np.nan
                file_top_desc = ['']
            #colors
            file_colors = file['imagePropertiesAnnotation']['dominantColors']['colors']            
            file_color_score = np.asarray([x['score'] for x in file_colors]).mean()
            file_color_pixelfrac = np.asarray([x['pixelFraction'] for x in file_colors]).mean()            
            #crops
            file_crops = file['cropHintsAnnotation']['cropHints']                
            file_crop_conf = np.asarray([x['confidence'] for x in file_crops]).mean()
            if 'importanceFraction' in file_crops[0].keys():
                file_crop_importance = np.asarray([x['importanceFraction'] for x in file_crops]).mean()
            else:
                file_crop_importance = np.nan
    
                
            new = pd.DataFrame(
                    {
                            'PetID'          : [i],
                            'annots_score'   : [file_top_score],
                            'color_score'     : [file_color_score],
                            'color_pixelfrac' : [file_color_pixelfrac],
                            'crop_conf'       : [file_crop_conf],
                            'crop_importance' : [file_crop_importance],
                            'annots_top_desc' : [' '.join(file_top_desc)]})
            feature_metadata_sub = feature_metadata_sub.append(new)
                
        metadata_desc = feature_metadata_sub.groupby(['PetID'])['annots_top_desc'].unique()
        metadata_desc = metadata_desc.reset_index()
        metadata_desc['annots_top_desc'] = metadata_desc['annots_top_desc'].apply(lambda x:' '.join(x))
        feature_metadata_sub.drop(['annots_top_desc'], axis=1, inplace=True)
            
        feature_metadata_sub = feature_metadata_sub.groupby(['PetID']).agg(['mean', 'sum'])
        feature_metadata_sub.columns = ['{}_{}'.format(c[0], c[1].upper()) for c in feature_metadata_sub.columns.tolist()]  
        feature_metadata_sub = feature_metadata_sub.reset_index()
            
        feature_metadata = feature_metadata.append(feature_metadata_sub)
    return feature_metadata


#
    

#for each in 
#train_feature_metadata = extract_metadata_feature('fffd78a11-1', 'train')

train_feature_metadata = Parallel(n_jobs=8, verbose=1)(
        delayed(extract_metadata_feature)(i, 'train') for i in train.PetID)
train_feature_metadata = [x for x in train_feature_metadata]
train_feature_metadata = pd.concat(train_feature_metadata, ignore_index=True, sort=False)

test_feature_metadata = Parallel(n_jobs=8, verbose=1)(
        delayed(extract_metadata_feature)(i, 'test') for i in test.PetID)
test_feature_metadata = [x for x in test_feature_metadata]
test_feature_metadata = pd.concat(test_feature_metadata, ignore_index=True, sort=False)

#%% 连接sentiment和metadata和原始数据
x_train = df_train.merge(train_feature_sentiment, how='left', on='PetID')
x_train = x_train.merge(train_feature_metadata, how='left', on='PetID')

y_train = x_train['AdoptionSpeed']
x_train.drop(['AdoptionSpeed'], axis=1, inplace=True)

x_test = df_test.merge(test_feature_sentiment, how='left', on='PetID')
x_test = x_test.merge(test_feature_metadata, how='left', on='PetID')

#%% RescuerID 处理
df = df_train.append(df_test)
data_rescuer = df.groupby(['RescuerID'])['PetID'].count().reset_index()
data_rescuer.columns = ['RescuerID', 'RescuerID_count']
data_rescuer['Rescuer_count_rank'] = data_rescuer['RescuerID_count'].rank(pct=True)

x_train = x_train.merge(data_rescuer, how='left', on='RescuerID')
x_test  = x_test.merge(data_rescuer, how='left', on='RescuerID')




#%% 处理breed 
# 是非有第二血统
x_train['HasSecondBreed'] = x_train['Breed2'].map(lambda x:1 if x != 0 else 0)
x_test['HasSecondBreed'] = x_test['Breed2'].map(lambda x:1 if x != 0 else 0)

#train_breed_main = x_train[['Breed1']].merge(
#    labels_breed, how='left',
#    left_on='Breed1', right_on='BreedID',
#    suffixes=('', '_main_breed'))
#
#train_breed_main = train_breed_main.iloc[:, 2:]
#train_breed_main = train_breed_main.add_prefix('main_breed_')
#
#train_breed_second = x_train[['Breed2']].merge(
#    labels_breed, how='left',
#    left_on='Breed2', right_on='BreedID',
#    suffixes=('', '_second_breed'))
#
#
#train_breed_second = train_breed_second.iloc[:, 2:]
#train_breed_second = train_breed_second.add_prefix('second_breed_')
#
#x_train = pd.concat(
#    [x_train, train_breed_main, train_breed_second], axis=1)
#
###############
#test_breed_main = x_test[['Breed1']].merge(
#    labels_breed, how='left',
#    left_on='Breed1', right_on='BreedID',
#    suffixes=('', '_main_breed'))
#
#test_breed_main = test_breed_main.iloc[:, 2:]
#test_breed_main = test_breed_main.add_prefix('main_breed_')
#
#test_breed_second = x_test[['Breed2']].merge(
#    labels_breed, how='left',
#    left_on='Breed2', right_on='BreedID',
#    suffixes=('', '_second_breed'))
#
#test_breed_second = test_breed_second.iloc[:, 2:]
#test_breed_second = test_breed_second.add_prefix('second_breed_')
#
#x_test = pd.concat(
#    [x_test, test_breed_main, test_breed_second], axis=1)
#
#print(x_train.shape, x_test.shape)
#
#categorical_columns = ['main_breed_BreedName', 'second_breed_BreedName']
#for i in categorical_columns:
#    x_train.loc[:, i] = pd.factorize(x_train.loc[:, i])[0]

#%% 数据清理
drop_columns = ['Name', 'RescuerID', 'Description', 'PetID', 'token']
col = ['sentence_magnitude', 'sentence_score', 'document_magnitude', 'document_score']

x_train.drop(drop_columns, axis=1, inplace=True)
x_test.drop(drop_columns, axis=1, inplace=True)

x_train[col] = x_train[col].astype(float)
x_test[col]  = x_test[col].astype(float)

#%% lgb
from lightgbm.sklearn import LGBMClassifier



model_lgb = LGBMClassifier(
        learning_rate    = 0.1,
        n_estimators     = 500,
        max_depth        = 7,
        num_leaves       = 20,
        subsample        = 0.9,
        feature_fraction = 0.7,
        n_jobs           = -1,
        random_state     = 40
        )
        
model_lgb.fit(x_train, y_train)


#%%
#model_xgb.predict(x_test)
y_pre = model_lgb.predict(x_train)
val = cross_val_score(model_lgb, x_train, y_train, scoring = scorer, cv=3).mean()








