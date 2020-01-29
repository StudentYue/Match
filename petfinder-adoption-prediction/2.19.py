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

#from joblib import Parallel, delayed
from tqdm import tqdm
from PIL import Image

#%%
df_train  = pd.read_csv('train.csv')
df_test   = pd.read_csv('test.csv')

train = df_train.copy()
test  = df_test.copy()

#%% 提取 sentiment 的特征
# train
def extract_sentiment_feature(x):
    i = 1
    
    feature_sentiment = pd.DataFrame(columns=['PetID', 'token', 'sentence_magnitude', 'sentence_score','document_magnitude', 'document_score'])

    if x == 'train':
        set_df = df_train
        set_file = 'train'
    else:
        set_df = df_test
        set_file = 'test'
    
    for each in set_df.PetID[:5]:
        i += 1
        print(i)
        
        file_name = '{}_sentiment/{}.json'.format(set_file,each)
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
                    {'PetID':[each], 
                     'token'               : [token],
                     'sentence_magnitude'  : [sentenceSentiment_magnitude],
                     'sentence_score'      : [sentenceSentiment_score],
                     'document_magnitude'  : [docementSentiment_magnitude], 
                     'document_score'      : [documentSentiment_score]})  
            feature_sentiment = feature_sentiment.append(new)
        except:
            print('{}没找到'.format(file_name))
    return feature_sentiment

#
train_feature_sentiment = extract_sentiment_feature('train')
test_feature_sentiment  = extract_sentiment_feature('test')

#%% 提取 metadata 的特征
#file_name = 'train_metadata/000a290e4-1.json'
#f = open(file_name, 'r')
#metadatafile = json.load(f)

def extract_metadata_feature(x):    
    i = 1
    
    feature_metadata = pd.DataFrame()
    if x == 'train':
        set_df = df_train
        set_file = 'train'
    else:
        set_df = df_test
        set_file = 'test'

    for each in set_df.loc[:,'PetID']:
        i += 1
        print(i)
        
        metadata_filenames = sorted(glob.glob('{}_metadata/{}*.json'.format(set_file, each)))
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
                                'PetID'          : [each],
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
    


train_feature_metadata = extract_metadata_feature('train')
test_feature_metadata  = extract_metadata_feature('test')




























