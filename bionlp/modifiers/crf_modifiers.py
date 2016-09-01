import logging,nltk
from tqdm import tqdm
logger=logging.getLogger(__name__)

from copy import deepcopy
from bionlp.data.token import Token as Token
from bionlp.data.sentence import Sentence as Sentence
from bionlp.data.document import Document as Document
from bionlp.data.dataset import Dataset as Dataset

import modifier_utils

def add_POS(dataset):
    dataset.active.append('POS')
    for document in tqdm(dataset.value):
        for sent in document.value:
            word_list= sent.get_list()
            pos_tags=nltk.pos_tag(word_list)
            for word_id,token in enumerate(sent.value):
                assert token.value == pos_tags[word_id][0],'Alignment mismatch in POS tagging'
                token.attr['POS']=pos_tags[word_id][1]
    return dataset



def add_sentiment(dataset,objectivity=None):
    if objectivity==1:
        attr="-objective"
    else:
        attr=""
    dataset.active.append('positive'+attr)
    dataset.active.append('negative'+attr)
    for document in tqdm(dataset.value):
        for sent in document.value:
            for word_id,token in enumerate(sent.value):
                pos,neg=modifier_utils.get_avg_sentiment(token.value,objectivity)
                token.attr['positive'+attr]=pos
                token.attr['negative'+attr]=neg
    return dataset

def trim_tags(dataset):
    # Specific to UMassMed dataset
    dataset.passive.append('trim-tags')
    for document in tqdm(dataset.value):
        for sent in document.value:
            for word_id,token in enumerate(sent.value):
                if token.attr['Annotation'] =='ADE+occured' or token.attr['Annotation'] =='adverse+effect':
                    sent.value[word_id].attr['Annotation']='ADE'
                if token.attr['Annotation'] == 'MedDRA':
                    sent.value[word_id].attr['Annotation']='None'
    return dataset



def add_BIO(dataset):
    exempt_list=['ADE'] # BIO tags are not applied to these labels, because they are already too few. 
    logger.info('Exempt list is {0}'.format(exempt_list))
    dataset.passive.append('BIO')
    for document in tqdm(dataset.value):
        for sent in document.value:
            sentence= deepcopy(sent.value)
            for word_id,token in enumerate(sentence):
                if (word_id==0 or sentence[word_id-1].attr['Annotation']!=sentence[word_id].attr['Annotation']) and sentence[word_id].attr['Annotation'] not in exempt_list:
                    sent.value[word_id].attr['Annotation']='B-'+token.attr['Annotation']
    return dataset


def add_delayed_modifiers(dataset,modifier):
    # provided for supporting future modifications
    dataset.delayed.append(modifier)
    return dataset
