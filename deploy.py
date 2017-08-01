import sys
import os
import json
import pickle
import logging
from analysis import io
from bionlp.taggers.rnn_feature.tagger import rnn_train
from bionlp.preprocess.dataset_preprocess import encode_data_format, decode_training_data
from bionlp.preprocess.extract_data import get_text_from_files
from bionlp.modifiers.crf_modifiers import add_BIO
from bionlp.modifiers.rnn_modifiers import add_surface_feature_list, add_umls_type, construct_umls_rnn_features
from bionlp.utils.crf_arguments import deploy_arguments
from bionlp.evaluate.evaluation import get_Approx_Metrics, get_Exact_Metrics, strip_bio, create_confusion_matrix
from bionlp.evaluate.postprocess import prepare_document_report


logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# this is used to treat bioNLP as a local python pack
full_path = os.path.realpath(__file__)
sys.path.insert(0, full_path)


def trainer(params):
    logger.info('Loading new input dataset from  {0}'.format(deploy_params['input']))
    if not params['noeval']:
        logger.info('Evaluation on the input dataset is on ( -noeval 0). '
                    'I will search for gold standard annotation in .json files ')
    raw_text, documents = get_text_from_files(deploy_params['input'], params['umls'],
                                              include_annotations=not params['noeval'])
    encoded_documents = encode_data_format(documents, raw_text, params['umls'])

    if params['umls'] != 0 and 'UMLS_TYPE' not in encoded_documents.active:
        logger.info('UMLS is on, but the dataset does not have umls tags. Adding and caching ..')
        encoded_documents = add_umls_type(encoded_documents)
    del raw_text
    del documents
    logger.info('Gathering stored vocabularies')
    model_file = pickle.load(open(params['model'], 'rb'), encoding='latin1')
    w2i = model_file['w2i']
    umls_vocab_dict = model_file['umls_vocab']

    encoded_documents = add_surface_feature_list(encoded_documents)
    if params['umls'] != 0:
        encoded_documents, umls_vocab_dict = construct_umls_rnn_features(encoded_documents, umls_vocab_dict)
    logger.info('Dataset Tags \nActive {0}\nPassive {1}\nDelayed {2}'.format(
        encoded_documents.active, encoded_documents.passive, encoded_documents.delayed))

    if params['bio'] == 1:
        encoded_documents = add_BIO(encoded_documents)

    # Decoding Data into training format #
    dataset = decode_training_data(encoded_documents)

    texts, label, pred = rnn_train(dataset, params, w2i, umls_vocab_dict)
    if params['output-dir'] is not 'None':
        prepare_document_report(texts, label, pred, encoded_documents, params['output-dir'])
    if params['eval-file'] is not 'None':
        true, predicted = strip_bio(label, pred)
        cm = create_confusion_matrix(true, predicted)
        io.pickle_confusion_matrix(cm, params['eval-file'])
    if not params['noeval']:
        get_Exact_Metrics(label, pred)
        get_Approx_Metrics(label, pred)


if __name__ == "__main__":
    deploy_params = deploy_arguments()

    # check
    # http://stackoverflow.com/questions/28218466/unpickling-a-python-2-object-with-python-3
    nn_v_d = pickle.load(open(deploy_params['model'], 'rb'), encoding='latin1')

    config_params = nn_v_d['params']
    config_params['model'] = deploy_params['model']
    config_params['eval-file'] = deploy_params['eval-file']
    config_params['noeval'] = deploy_params['noeval']
    config_params['output-dir'] = deploy_params['outputdir']
    config_params['error-analysis'] = deploy_params['output']
    config_params['dependency'] = {}
    config_params['trainable'] = False
    # Using hardcoded dataset percentage
    config_params['dataset_percentage'] = 100

    print("Using the parameters :")
    print(json.dumps(config_params, indent=2))
    trainer(config_params)
