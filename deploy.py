import sys,os,json,pickle
full_path = os.path.realpath(__file__)
sys.path.insert(0,full_path)

###### code above this line is used to treat bioNLP as a local python package. #####

import logging
logging.basicConfig(level=logging.INFO)
logger=logging.getLogger(__name__)


from bionlp.taggers.rnn_feature.tagger import rnn_train
from bionlp.preprocess.dataset_preprocess import encode_data_format,decode_training_data
from bionlp.preprocess.extract_data import file_extractor,annotated_file_extractor
from bionlp.data.token import Token as Token
from bionlp.data.sentence import Sentence as Sentence
from bionlp.data.document import Document as Document
from bionlp.data.dataset import Dataset as Dataset
from bionlp.modifiers.crf_modifiers import add_BIO
from bionlp.modifiers.rnn_modifiers import add_surface_feature_list,add_umls_type,construct_umls_rnn_features
from bionlp.utils.crf_arguments import deploy_arguments
from bionlp.evaluate.evaluation import get_Approx_Metrics,get_Exact_Metrics,evaluator
from bionlp.evaluate.postprocess import prepare_document_report

def trainer(config_params):
    logger.info('Loading new input dataset from  {0}'.format(deploy_params['input']))
    if not config_params['noeval']:
        logger.info('Evaluation on the input dataset is on ( -noeval 0). I will search for gold standard annotation in .json files ')
        raw_text,documents=annotated_file_extractor(deploy_params['input'],config_params['umls'])
    else:
        raw_text,documents=file_extractor(deploy_params['input'],config_params['umls'])
    encoded_documents=encode_data_format(documents,raw_text,config_params['umls'])

    if config_params['umls']!=0 and 'UMLS_TYPE' not in encoded_documents.active:
        logger.info('UMLS is on, but the dataset does not have umls tags. Adding and caching ..')
        encoded_documents=add_umls_type(encoded_documents)
    del raw_text
    del documents
    logger.info('Gathering stored vocabularies')
    model_file= pickle.load(open(config_params['model'],'rb'))
    w2i=model_file['w2i']
    umls_vocab_dict=model_file['umls_vocab']


    encoded_documents=add_surface_feature_list(encoded_documents)
    if config_params['umls']!=0:
        encoded_documents,umls_vocab_dict=construct_umls_rnn_features(encoded_documents,umls_vocab_dict)
    logger.info('Dataset Tags \nActive {0}\nPassive {1}\nDelayed {2}'.format(encoded_documents.active,encoded_documents.passive,encoded_documents.delayed))

    #logger.info('Sample of encoded data {0}'.format(encoded_documents.value[0].value[1].value[0]))
    if config_params['bio']==1:
        encoded_documents=add_BIO(encoded_documents)

    ##### Decoding Data into training format #####
    dataset=decode_training_data(encoded_documents)
    #logger.info('Sample of decoded data {0}'.format(dataset[0][2]))
    texts,label,pred=rnn_train(dataset,config_params,w2i,umls_vocab_dict)
    if config_params['output-dir'] is not 'None':
        prepare_document_report(texts,label,pred,encoded_documents,config_params['output-dir'])
    if not config_params['noeval']:
        evaluator(label,pred,get_Exact_Metrics)
        evaluator(label,pred,get_Approx_Metrics)

def main(config_params):

    config_params['dependency']={}

    config_params['trainable']=False
    config_params['dataset_percentage']=100            # Using hardcoded dataset percentage
    config_params['error-analysis']=deploy_params['output']
    print "Using the parameters :"
    print json.dumps(config_params,indent =2)
    trainer(config_params)
    print "Using the parameters :"
    print json.dumps(config_params,indent =2)


if __name__=="__main__":
    deploy_params=deploy_arguments()
    nn_v_d=pickle.load(open(deploy_params['model'],'rb'))
    config_params=nn_v_d['params']
    config_params['model']=deploy_params['model']
    config_params['noeval']=deploy_params['noeval']
    config_params['output-dir']=deploy_params['outputdir']
    main(config_params)

