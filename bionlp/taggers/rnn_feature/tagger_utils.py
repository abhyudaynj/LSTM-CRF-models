import pickle,logging,random,gensim,os
import sklearn,pickle
from random import shuffle
from sklearn.metrics import f1_score,recall_score,precision_score
import collections
logger=logging.getLogger(__name__)
import numpy as np
import random
import string
from nltk.corpus import sentiwordnet as swn
from nltk.metrics import ConfusionMatrix


# Adapted from Keras pad and mask function
def pad_and_mask(X,U,Y,Z,maxlen ,padding='pre', value=0.):
    '''
        Override keras method to allow multiple feature dimensions.

        @dim: input feature dimension (number of features per timestep)
    '''
    lengths = [len(s) for s in Y]
    if maxlen is None:
        maxlen = np.max(lengths)
    y_dim =max([max(s) for s in Y])+1
    x_dim = 1
    u_dim = U[0][0].__len__()
    identity_mask=np.eye(y_dim)
    Y=[[identity_mask[w] for w in s] for s in Y]
    nb_samples = len(Y)
    logger.info('Maximum sequence length is {0}\nThe X dimension is {3}\nThe y dimension is {1}\nThe number of samples in dataset are {2}'.format(maxlen,y_dim,nb_samples,x_dim))

    x = np.zeros((nb_samples, maxlen,x_dim))
    u = np.zeros((nb_samples, maxlen,u_dim))
    y = (np.ones((nb_samples, maxlen, y_dim)) * value).astype('int32')
    mask = (np.ones((nb_samples, maxlen,)) * 0.).astype('int32')
    for idx, s in enumerate(X):
        X[idx]=X[idx][:maxlen]
        Y[idx]=Y[idx][:maxlen]
        U[idx]=U[idx][:maxlen]

    for idx, s in enumerate(X):
        if padding == 'post':
            x[idx, :len(X[idx]),0] = X[idx]
            y[idx, :len(X[idx])] = Y[idx]
            u[idx, :len(X[idx])] = U[idx]
            mask[idx, :len(X[i])] = 1
        elif padding == 'pre':
            x[idx, -len(X[idx]):,0] = X[idx]
            y[idx, -len(X[idx]):] = Y[idx]
            u[idx, -len(X[idx]):] = U[idx]
            mask[idx, -len(X[idx]):] = 1
    z=np.array(Z)
    logger.info('X shape : {0}\nY shape : {1}\nMask shape : {2}\nU shape : {3}\n Z shape {4}'.format(x.shape,y.shape,mask.shape,u.shape,z.shape))
    return x,u,y,z,mask

# added extra token_object iterable in hurry. TO DO : Refactor later.
def iterate_minibatches(inputs,mask,targets, batchsize,token_objects=None):
    useless_entries =0
    if token_objects != None:
        indices=np.array(range(len(inputs)))
        np.random.shuffle(indices)
        start_idx=0
        for start_idx in range(0, len(inputs) - batchsize + 1, batchsize):
            yield inputs[indices[start_idx:start_idx + batchsize]],mask[indices[start_idx:start_idx + batchsize]], targets[indices[start_idx:start_idx + batchsize]], token_objects[indices[start_idx:start_idx + batchsize]]
        if len(indices[start_idx+batchsize:]) >0:
            last_inputs=inputs[indices[-batchsize:]]
            last_targets=targets[indices[-batchsize:]]
            last_mask=mask[indices[-batchsize:]]
            last_token=token_objects[indices[-batchsize:]]
            #last_mask[:-len(indices[start_idx+batchsize:]),:-1]=0
            last_mask[:-len(indices[start_idx+batchsize:])]=0
            useless_entries+=len(last_mask[:-len(indices[start_idx+batchsize:])])
            yield last_inputs,last_mask,last_targets,last_token
    else:
        indices=np.array(range(len(inputs)))
        np.random.shuffle(indices)
        start_idx=0
        for start_idx in range(0, len(inputs) - batchsize + 1, batchsize):
            yield inputs[indices[start_idx:start_idx + batchsize]],mask[indices[start_idx:start_idx + batchsize]], targets[indices[start_idx:start_idx + batchsize]]
        if len(indices[start_idx+batchsize:]) >0:
            last_inputs=inputs[indices[-batchsize:]]
            last_targets=targets[indices[-batchsize:]]
            last_mask=mask[indices[-batchsize:]]
            #last_mask[:-len(indices[start_idx+batchsize:]),:-1]=0
            last_mask[:-len(indices[start_idx+batchsize:])]=0
            useless_entries+=len(last_mask[:-len(indices[start_idx+batchsize:])])
            yield last_inputs,last_mask,last_targets
        logger.info('{0} placeholder rows added to make sure that batch size is maintained'.format(useless_entries))



def get_vocab(tagged_data):
    vocab_set=set()
    tag_set=set()
    for i,sent in enumerate(tagged_data):
        vocab_set= vocab_set | set([x[0].lower() for x,y in tagged_data[i] if x[0] !='\x00'])
        tag_set = tag_set | set([y for x,y in tagged_data[i]])
    return vocab_set,tag_set


def trim_tags(tagged_data):
    #Function specific to UMass BioNLP data.
    for i,sent in enumerate(tagged_data):
        tagged_data[i]=[(x,'ADE') if y=='ADE+occured' or y=='adverse+effect' else (x,y) for x,y in tagged_data[i]]
        tagged_data[i]=[(x,'None') if y=='MedDRA' else (x,y) for x,y in tagged_data[i]]
    return tagged_data

def get_embedding_weights(w2i,params):
    i2w={i: word for word, i in w2i.iteritems()}
    logger.info('embedding sanity check (should be a word) :{0}'.format(i2w[12]))
    if params['word2vec']==1 and params['trainable']:
        if 'mdl' in params['dependency'] and os.path.isfile(params['dependency']['mdl']):
            mdl=gensim.models.Word2Vec.load_word2vec_format(params['dependency']['mdl'],binary=True)
            logger.info('{0},{1}'.format(mdl['is'].shape,len(w2i)))
        else:
            logger.warning('No word2vec model binary file found. Loading random weight vectors instead.')
            mdl={}
    else:
        # Use random initialization for embeddings, if word2vec option is 0 or if this is a deploy run. In deploy runs the relevant embeddings will be reset later.
        mdl={}
    emb_i=np.array([mdl[str(i2w[i])] if i in i2w and str(i2w[i]) in mdl else np.zeros(200,) for i in xrange(len(w2i))])
    return emb_i

def construct_binary_features(tagged_sentence):
    return [word[0][1].attr['SURFACE'] for word in tagged_sentence]


def encode_words(tagged_data,entire_note,params,vocab):
    shuffle(tagged_data)
    #flattening notes into sentences.
    if entire_note:             # governed by params['document']. This feature is off always for CRF decoding
        note_data=[]
        for notes in tagged_data:
            note=[word for sent in notes for word in sent+[(('EOS_',{'POS':'EOS_'}),'None')]]
            note_data.append(note)
        tagged_data=note_data
    else:
        tagged_data=[sentence for notes in tagged_data for sentence in notes]
    tagged_data=trim_tags(tagged_data)

    # splitting sentences larger than maxlen into two. 
    split_tagged_data=[]
    for idxs,sente in enumerate(tagged_data):
        if sente.__len__() > params['maxlen']:
            split_sente=[sente[idx*params['maxlen']:(idx+1)*params['maxlen']] for idx,_ in enumerate(sente[::params['maxlen']])]
            split_tagged_data.extend(split_sente)
        else:
            split_tagged_data.append(sente)

    tagged_data=split_tagged_data

    v_set,t_set=get_vocab(tagged_data)
    if 'trainable' in params and params['trainable'] == False and params['model']!='None':
        logger.info('Trainable is off and valid model filename is provided. Reading and storing the word and tag list order from model file {0}'.format(params['model']))
        nn_dict=pickle.load(open(params['model'],'rb'))
        w2i=nn_dict['w2i']
        t2i=nn_dict['t2i']
        emb_w=nn_dict
    else:
        if vocab is not None:
            logger.info('Vocab already provided. I will load using that')
            w2i=vocab
            logger.info('Total Word Vocabulary Size : {0}'.format(len(w2i)))
        else:
            w2i={word :i+1 for i,word in enumerate(list(v_set))}
            w2i['OOV_CHAR']=0
        t2i={word :i for i,word in enumerate(list(t_set))}
    logger.info('embedding sanity check (should be a number >1):{0}'.format(w2i['is']))
    X=[None]*len(tagged_data)
    Y=[None]*len(tagged_data)
    Z=[None]*len(tagged_data)
    U=[None]*len(tagged_data)
    logger.info('Preparing data ...')
    label_count=len(t2i)
    for i,sent in enumerate(tagged_data):
        x=[w2i[word.lower()] if word.lower() in w2i else 0 for (word,tag),label in tagged_data[i]]
        U[i]=construct_binary_features(tagged_data[i])
        z=[token_object for (word,token_object),label in tagged_data[i]]
        if 'trainable' in params and params['trainable'] == False and 'noeval' in params and params['noeval']==True:
            y=[ix % label_count for ix,(word,label) in enumerate(tagged_data[i])]  
            # trick to make sure that y dimesions are same for deployement. This is needed, because we use y_in multiple times while constructing the symbolic functions
        else:
            y=[t2i[label] if label in t2i else 0 for word,label in tagged_data[i]]
        X[i]=x
        Y[i]=y
        Z[i]=z
    emb_w=get_embedding_weights(w2i,params)
    return X,U,Z,Y,label_count,emb_w,t2i,w2i


def load_data(dataset,params,nb_words=None, test_split=0.2,entire_note=False,vocab=None):
    original_tokens = sum([sum([len(sente_) for sente_ in doc_]) for doc_ in dataset])
    logger.info('original token {0}'.format(dataset[0][0][0]))
    X,U,Z,Y,numTags,emb_w,t2i,w2i =  encode_words(dataset,entire_note,params,vocab)
    maxlen=params['maxlen']

    if maxlen:
        logger.info('Truncating {0} instances out of {1}'.format(sum(1 if len(y)>100 else 0 for y in Y),sum(1 for y in Y)))

    #logger.info('Final Format for NN processing\nx{0}\ny{1}\nz{2}\nu{3}'.format(X[0][:10],Y[0][:10],Z[0][:10],U[0][:10]))

    processed_tokens = sum([len(inst_) for inst_ in X])
    logger.info('processed token{0}'.format(X[0][0]))
    logger.info('Original and processed dataset token lengths are {0} and {1}'.format(original_tokens,processed_tokens))
    return (X,U,Z,Y),numTags,emb_w,t2i,w2i


