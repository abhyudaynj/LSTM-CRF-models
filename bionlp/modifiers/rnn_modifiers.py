import logging,nltk,itertools
from tqdm import tqdm
import numpy as np
import json
from operator import itemgetter
logging.basicConfig(level=logging.INFO)
logger=logging.getLogger(__name__)


from bionlp.data.token import Token as Token
from bionlp.data.sentence import Sentence as Sentence
from bionlp.data.document import Document as Document
from bionlp.data.dataset import Dataset as Dataset
from modifier_utils import punct_list

def add_umls_type(dataset,cache_file=None):
    dataset.active.append('UMLS_TYPE')
    logger.info('Adding UMLS Semantic Type information')
    for current_doc in tqdm(dataset.value):
        text_i=current_doc.attr['raw_text']
        tag_positions=[[] for idxs in xrange(len(text_i))]
        umls_objects=current_doc.attr['metamap_anns']
        sorted_umls_objects=sorted(umls_objects,key=itemgetter('begin'))
        for umls_obj in umls_objects:
            for idx in xrange(umls_obj['begin'],umls_obj['end']):
                tag_positions[idx].append(umls_obj['sem_type'].split('+'))

        for sent in current_doc.value:
            for word in sent.value:
                t_list=list(itertools.chain.from_iterable(tag_positions[word.attr['offset']:word.attr['offset']+len(word.value)]))
                t_list=list(itertools.chain.from_iterable(t_list))
                type_list=set(t_list)
                word.attr['umls_type']=list(type_list)

    return dataset


def construct_umls_rnn_features(dataset,umls_type_vocab=None):
    # For now this modifier should always be called after add_surface_feature_list()
    assert 'SURFACE' in dataset.value[0].value[0].value[0].attr,'add_surface_feature_list() should be called before construct_umls_rnn_features()'
    if umls_type_vocab is None:
        logging.info('Vocabulary for semantic types not provided. Scanning for all UMLS Semantic Types')
        type_vocab=set()
        for document in tqdm(dataset.value):
            for sent in document.value:
                for word_id,token in enumerate(sent.value):
                    type_vocab =type_vocab.union(set(token.attr['umls_type']))
        type_vocab={umty:idxs for idxs,umty in enumerate(list(type_vocab))}
    else:
        logging.info('Using vocabulary for UMLS semantic type that was provided in dataset object')
        type_vocab=umls_type_vocab

    logger.info('Constructing multi-hot vectors for UMLS features')
    logger.info('UMLS Vocab {0}'.format(type_vocab))
    vocab_size=len(type_vocab)
    logger.info('UMLS type vocab size {0}'.format(vocab_size))
    for document in tqdm(dataset.value):
        for sent in document.value:
            for word_id,token in enumerate(sent.value):
                feature_list=[0]*vocab_size
                for umty in token.attr['umls_type']:
                    feature_list[type_vocab[umty]]=1
                token.attr['SURFACE'].extend(feature_list)
    return dataset,type_vocab


def add_surface_feature_list(dataset):
    dataset.active.append('SURFACE')
    logger.info('Adding Surface features to dataset')
    for document in tqdm(dataset.value):
        for sent in document.value:
            for word_id,token in enumerate(sent.value):
                feature_list=[]
                if token.value in punct_list:
                    feature_list.append(1)
                else:
                    feature_list.append(0)
                if token.value[0].isupper():
                    feature_list.append(1)
                else:
                    feature_list.append(0)

                if all([x.isupper() for x in token.value]):
                    feature_list.append(1)
                else:
                    feature_list.append(0)

                if any([x.isupper() for x in token.value]):
                    feature_list.append(1)
                else:
                    feature_list.append(0)

                if any([x.isdigit() for x in token.value]):
                    feature_list.append(1)
                else:
                    feature_list.append(0)
                token.attr['SURFACE']=feature_list

    return dataset


