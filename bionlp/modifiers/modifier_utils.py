from nltk.corpus import sentiwordnet as swn
import numpy as np
import json
from tqdm import tqdm
import logging,json
logger=logging.getLogger(__name__)


import string
punct_list=[x for x in string.punctuation if x!='.']


def get_avg_sentiment(word,objectivity=None):
    syns=swn.senti_synsets(word)
    pos=0.0
    neg=0.0
    if syns:
        if objectivity ==1:
            pos=np.mean([x._pos_score*x._obj_score for x in syns])
            neg=np.mean([x._neg_score*x._obj_score for x in syns])
        else:
            pos=np.mean([x._pos_score for x in syns])
            neg=np.mean([x._neg_score for x in syns])
    return (pos,neg)

def remove_tags(dataset,removed_list):
    logger.info('Removing the following tags from the dataset and replacing with None : {0}'.format(removed_list))
    dataset.passive.append('removed-tags')
    for document in tqdm(dataset.value):
        for sent in document.value:
            for word_id,token in enumerate(sent.value):
                if token.attr['Annotation'] in removed_list:
                    sent.value[word_id].attr['Annotation']='None'
    return dataset



