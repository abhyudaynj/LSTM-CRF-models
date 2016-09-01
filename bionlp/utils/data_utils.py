import numpy as np
import logging
logging.basicConfig(level=logging.INFO)
logger=logging.getLogger(__name__)


from bionlp.data.token import Token as Token
from bionlp.data.sentence import Sentence as Sentence
from bionlp.data.document import Document as Document
from bionlp.data.dataset import Dataset as Dataset

def get_emb_vocab(dataset_list):
    emb_w=set()
    for dataset in dataset_list:
        emb_d=[]
        for document in dataset.value:
            for sentence in document.value:
                for tkn in sentence.value:
                    emb_d.append(tkn.value.lower())
        emb_w=emb_w.union(emb_d)
        w2i={word :i+1 for i,word in enumerate(list(emb_w))}
        w2i['OOV_CHAR']=0
    return w2i


def make_cross_validation_sets(data_len,n,training_percent=None):
    if training_percent ==None:
        training_percent = 1.0
    else:
        training_percent = float(training_percent)/100.0
    split_length=int(data_len/n)
    splits=[None]*n
    for i in xrange(n):
        arr=np.array(range(data_len))
        test=range(i*split_length,(i+1)*split_length)
        mask=np.ones(arr.shape,dtype=bool)
        mask[test]=0
        train=arr[mask]
        training_len=float(len(train))*training_percent
        train=train[:int(training_len)]
        splits[i]=(train.tolist(),test)
    return splits


