from __future__ import print_function

import logging
logging.basicConfig(level=logging.INFO)
logger=logging.getLogger(__name__)

from tqdm import tqdm
from bionlp.data.token import Token as Token
from bionlp.data.sentence import Sentence as Sentence
from bionlp.data.document import Document as Document
from bionlp.data.dataset import Dataset as Dataset

def encode_data_format(documents,raw_text,umls_params):
    logger.info('Encoding dataset into data format')
    document_dict={did_:(dtext_,metamap_) for (did_,dtext_,metamap_) in raw_text}
    documentList=[]
    for did,document in tqdm(documents):
        sentenceList=[]
        for sid,sent in enumerate(document):
            tid=0
            tokenList=[]
            for token in sent:
                newToken=Token(token[0],tid)
                newToken.attr['Annotation']=token[2]
                newToken.attr['offset']=token[1]
                newToken.attr['document']=did
                tokenList.append(newToken)
                tid+=1
            newSentence=Sentence(tokenList,sid)
            sentenceList.append(newSentence)
        newDocument=Document(sentenceList,did)
        newDocument.attr['raw_text']=document_dict[did][0]
        if umls_params!=0:
            newDocument.attr['metamap_anns']=document_dict[did][1]
        documentList.append(newDocument)
    dataset =Dataset(documentList)
    dataset.passive.append('Annotation')
    return dataset

def decode_training_data(dataset):
    documentList=[]
    for document in dataset.value:
        sentenceList=[]
        for sent in document.value:
            sentenceList.append([((token.value,token),token.attr['Annotation']) for token in sent.value])
        documentList.append(sentenceList)
    logger.info('Number of Records decoded into training data format {0}'.format(documentList.__len__()))
    return documentList

def decode_n_strip_training_data(dataset):
    #Used for decoding the training data if required for Neural Network Models. 
    documentList=[]
    for document in dataset.value:
        sentenceList=[]
        for sent in document.value:
            sentenceList.append([(token.value,token.attr['Annotation']) for token in sent.value])
        documentList.append(sentenceList)
    logger.info('Number of Records decoded into training data format {0}'.format(documentList.__len__()))
    return documentList



