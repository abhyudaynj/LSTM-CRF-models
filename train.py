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

import scripts.utils as utilscripts
from bionlp.taggers.rnn_feature.tagger import rnn_train
from bionlp.preprocess.dataset_preprocess import encode_data_format, decode_training_data
from bionlp.preprocess.extract_data import get_text_from_files
from bionlp.modifiers.crf_modifiers import add_BIO
from bionlp.modifiers.rnn_modifiers import add_surface_feature_list, add_umls_type, construct_umls_rnn_features
from bionlp.evaluate.evaluation import get_Approx_Metrics, get_Exact_Metrics, evaluator


def load_label_blacklist(label_blacklist_file):
    if label_blacklist_file is not 'None':
        assert os.path.isfile(label_blacklist_file), "The label_blacklist_file '%s' can not be found or opened!" % label_blacklist_file
        with open(label_blacklist_file, "r") as blacklist_f:
            label_blacklist = [tag.strip() for tag in blacklist_f.read().split(',')]
    else:
        label_blacklist = []
    return label_blacklist


def trainer(config_params):
    if config_params['data-refresh'] == 1:
        logger.info('Loading new input dataset from  {0}'.format(config_params['input']))
        raw_text, documents = get_text_from_files(config_params['input'], config_params['umls'], include_annotations=True)

        label_blacklist = load_label_blacklist(config_params['label-blacklist-file'])

        encoded_documents = encode_data_format(documents, raw_text, config_params['umls'],
            config_params['sent-limit'], label_blacklist)
    else:
        encoded_documents = pickle.load(
            open(config_params['dependency']['data'], 'rb'), encoding='latin1')
        logger.info('Loaded preprocessed input dataset.\nNumber of documents extracted {0}'.format(
            encoded_documents.value.__len__()))

    w2i = None
    # ADDING extra vocab terms to NN embedding
    if config_params['extra-vocab'] > 0 and 'vocab-data' in config_params['dependency']:
        logger.info('Calculating Extra vocab from {0}'.format(
            config_params['dependency']['vocab-data']))
        extra_data = [pickle.load(open(dset, 'rb'), encoding='latin1')
                      for dset in config_params['dependency']['vocab-data']]
        logger.info('Loaded cached vocab dataset.')

        total_corpus = [encoded_documents] + extra_data
        w2i = utilscripts.get_emb_vocab(total_corpus)
    elif config_params['extra-vocab'] < 0 and 'mdl' in config_params['dependency'] and os.path.isfile(config_params['dependency']['mdl']):
        logger.info(
            'Loading the entire vocabulary of the word2vec file into the model')
        w2i = utilscripts.get_all_vocab(config_params['dependency']['mdl'])

    if config_params['umls'] != 0:
        encoded_documents = add_umls_type(encoded_documents)
    encoded_documents = add_surface_feature_list(encoded_documents)
    umls_vocab_dict = None
    if config_params['umls'] != 0:
        encoded_documents, umls_vocab_dict = construct_umls_rnn_features(
            encoded_documents)
        logger.info(
            'Adding extra 20 dimensions to enable extra features for UMLS')
        # Adding 20 extra feature dimensions to accomodate UMLS semantic types
        config_params['feature1'] += 20
    extra_data = []
    logger.info('Dataset Tags \nActive {0}\nPassive {1}\nDelayed {2}'.format(
        encoded_documents.active, encoded_documents.passive, encoded_documents.delayed))
    logger.info('Sample of encoded data {0}'.format(
        encoded_documents.value[0].value[1].value[0]))
    if config_params['bio'] == 1:
        encoded_documents = add_BIO(encoded_documents)

    ##### Decoding Data into training format #####
    dataset = decode_training_data(encoded_documents)
    logger.info('Sample of decoded data {0}'.format(dataset[0][2]))
    texts, label, pred = rnn_train(
        dataset, config_params, w2i, umls_vocab_dict)
    if config_params['deploy'] == 0:
        evaluator(label, pred, get_Exact_Metrics)
        evaluator(label, pred, get_Approx_Metrics)


def main(config_params):
    # loading data sources from dependency json
    datas = {}
    if os.path.isfile('dependency.json'):
        datas = json.load(open('dependency.json', 'r'))

    config_params['dependency'] = datas
    config_params['trainable'] = True
    # Using hardcoded dataset percentage
    config_params['dataset_percentage'] = 100
    print("Using the parameters :")
    print(json.dumps(config_params, indent=2))

    trainer(config_params)
