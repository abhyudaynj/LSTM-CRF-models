from __future__ import print_function

import logging,gensim
from tqdm import tqdm
logger=logging.getLogger(__name__)


from bionlp.data.token import Token as Token
from bionlp.data.sentence import Sentence as Sentence
from bionlp.data.document import Document as Document
from bionlp.data.dataset import Dataset as Dataset

def get_emb_vocab(dataset_list):
    logger.info('Creating vocabulary list from provided and extra datasets')
    emb_w=set()
    for dataset in dataset_list:
        emb_d=[]
        for document in tqdm(dataset.value):
            for sentence in document.value:
                for tkn in sentence.value:
                    emb_d.append(tkn.value.lower())
        emb_w=emb_w.union(emb_d)
    w2i={word :i+1 for i,word in enumerate(list(emb_w))}
    w2i['OOV_CHAR']=0
    logger.info('Total Vocabulary Size {0}'.format(len(w2i)))
    return w2i

def get_all_vocab(wordvec_model):
    logger.info('Loading the Entire vocabulary in Word Vector model file {0}'.format(wordvec_model))
    mdl=gensim.models.Word2Vec.load_word2vec_format(wordvec_model,binary=True)
    w2i={word :i+1 for i,word in enumerate(mdl.vocab.keys())}
    w2i['OOV_CHAR']=0
    logger.info('Total Vocabulary Size {0}'.format(len(w2i)))
    return w2i


