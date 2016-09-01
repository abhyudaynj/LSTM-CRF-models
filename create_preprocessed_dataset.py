# Code for creating a preprocessed dataset

import sys,os,json,pickle
full_path = os.path.realpath(__file__)
sys.path.insert(0,full_path)

###### code above this line is used to treat bioNLP as a local python package. #####

import logging
logging.basicConfig(level=logging.INFO)
logger=logging.getLogger(__name__)

import scripts.utils as utilscripts
from bionlp.taggers.rnn_feature.tagger import rnn_train
from bionlp.preprocess.dataset_preprocess import encode_data_format,decode_training_data
from bionlp.preprocess.extract_data import annotated_file_extractor
from bionlp.data.token import Token
from bionlp.data.sentence import Sentence
from bionlp.data.document import Document
from bionlp.data.dataset import Dataset
from bionlp.modifiers.crf_modifiers import add_BIO
from bionlp.modifiers.rnn_modifiers import add_surface_feature_list,add_umls_type,construct_umls_rnn_features
from bionlp.evaluate.evaluation import get_Approx_Metrics,get_Exact_Metrics,evaluator

if os.path.isfile('dependency.json'):
    datas=json.load(open('dependency.json','r'))

logger.info('Loading new input dataset from  {0}'.format(datas['infile-list']))
raw_text,documents=annotated_file_extractor(datas['infile-list'],1)

encoded_documents=encode_data_format(documents,raw_text,1)
pickle.dump(encoded_documents,open('data/cache.pkl','wb'))

'''
encoded_documents=add_umls_type(encoded_documents)
encoded_documents=add_surface_feature_list(encoded_documents)
encoded_documents=construct_umls_rnn_features(encoded_documents)
pickle.dump(encoded_documents,open('data/temp.pkl','wb'))
'''
