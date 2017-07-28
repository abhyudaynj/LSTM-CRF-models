# Code for creating a preprocessed dataset

import sys
import os
import json
import pickle
full_path = os.path.realpath(__file__)
sys.path.insert(0, full_path)

###### code above this line is used to treat bioNLP as a local python pack

import logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

from bionlp.preprocess.dataset_preprocess import encode_data_format, decode_training_data
from bionlp.preprocess.extract_data import get_text_from_files


if os.path.isfile('dependency.json'):
    datas = json.load(open('dependency.json', 'r'))

logger.info('Loading new input dataset from  {0}'.format(datas['infile-list']))
raw_text, documents = get_text_from_files(datas['infile-list'], 1, include_annotations=True)

encoded_documents = encode_data_format(documents, raw_text, 1)
pickle.dump(encoded_documents, open('data/cache.pkl', 'wb'))

'''
encoded_documents=add_umls_type(encoded_documents)
encoded_documents=add_surface_feature_list(encoded_documents)
encoded_documents=construct_umls_rnn_features(encoded_documents)
pickle.dump(encoded_documents,open('data/temp.pkl','wb'))
'''
