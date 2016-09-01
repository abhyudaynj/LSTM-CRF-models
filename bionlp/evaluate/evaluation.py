import pickle,itertools,logging
import json
from nltk.metrics import ConfusionMatrix
from sklearn.metrics import f1_score,recall_score,precision_score
import collections

IGNORE_TAG='None'
logging.basicConfig(level=logging.INFO)
logger=logging.getLogger(__name__)

def get_labels(label,predicted):
    labels = list(set(itertools.chain.from_iterable(label)) | set(itertools.chain.from_iterable(predicted)))
    return labels

def get_Approx_Metrics(y_true,y_pred,verbose =True,preMsg='',flat_list=False):

    if verbose:
        print('------------------------ Approx Metrics---------------------------')
    if flat_list:
        z_true=y_true
        z_pred=y_pred
    else:
        z_true=list(itertools.chain.from_iterable(y_true))
        z_pred=list(itertools.chain.from_iterable(y_pred))
    z_true=[token[2:] if token[:2]=='B-' else token for token in z_true]
    z_pred=[token[2:] if token[:2]=='B-' else token for token in z_pred]
    label_dict={x:i for i,x in enumerate(list(set(z_true) | set(z_pred)))}
    freq_dict=collections.Counter(z_true)
    z_true=[ label_dict[x] for x in z_true]
    z_pred=[ label_dict[x] for x in z_pred]
    f1s= f1_score(z_true,z_pred, average=None)
    rs= recall_score(z_true,z_pred, average=None)
    ps= precision_score(z_true,z_pred, average=None)
    results =[]
    f1_none=[]
    avg_recall=avg_precision=avg_f1=0.0
    for i in label_dict:
            if verbose:
                print("{5} The tag \'{0}\' has {1} elements and recall,precision,f1 ={3},{4}, {2}".format(i,freq_dict[i],f1s[label_dict[i]],rs[label_dict[i]],ps[label_dict[i]],preMsg))
            if i!='None' and i!='|O':
                f1_none=f1_none+[(f1s[label_dict[i]],freq_dict[i]),]
                avg_recall+=float(rs[label_dict[i]])*float(freq_dict[i])
                avg_precision+=float(ps[label_dict[i]])*float(freq_dict[i])
    intermediate_sum=sum([float(z[1]) for z in f1_none])
    if intermediate_sum ==0:
        intermediate_sum +=1
    avg_recall =float(avg_recall)/float(intermediate_sum)
    avg_precision =float(avg_precision)/float(intermediate_sum)
    if (float(avg_recall)+float(avg_precision)) !=0.0:
        avg_f1=2.0*float(avg_precision)*float(avg_recall)/(float(avg_recall)+float(avg_precision))
    else:
        avg_f1=0.0
    if verbose:
        print("All medical tags collectively have {0} elements and recall,precision,f1 ={1},{2}, {3}".format(intermediate_sum,avg_recall,avg_precision,avg_f1))
    return avg_f1

def get_ConfusionMatrix(true,predicted):
    #Confusion Matrix is only valid for partial evaluation.
    true_chain=list(itertools.chain.from_iterable(true))
    predicted_chain=list(itertools.chain.from_iterable(predicted))
    print("Confusion Matrix of combined folds (partial evaluation)\n{0}".format(ConfusionMatrix(true_chain,predicted_chain)))


def get_Exact_Metrics(true,predicted,verbose=True):
    true,predicted=strip_BIO(true,predicted)
    if verbose:
        print('------------------------ Exact Metrics---------------------------')
        get_ConfusionMatrix(true,predicted)
    labels=get_labels(true,predicted)
    true_positive={label:0 for label in labels}
    trues={label:0 for label in labels}
    positives={label:0 for label in labels}
    for i,sent in enumerate(true):
        if sent.__len__() ==0:
            continue
        label_tags=[]
        predicted_tags=[]
        j=0
        tag='Nothing'
        pos=[]
        while j<len(sent):
            if tag!=sent[j]:
                if tag != 'Nothing':
                    label_tags.append((tag,tuple(pos)))
                pos=[]
                pos.append(j)
                tag=sent[j]
            else:
                pos.append(j)
            j+=1
        label_tags.append((tag,tuple(pos)))

        j=0
        tag='Nothing'
        pos=[]
        psent=predicted[i]
        while j<len(psent):
            if tag!=psent[j]:
                if tag != 'Nothing':
                    predicted_tags.append((tag,tuple(pos)))
                pos=[]
                pos.append(j)
                tag=psent[j]
            else:
                pos.append(j)
            j+=1
        predicted_tags.append((tag,tuple(pos)))
        for z in predicted_tags:
            positives[z[0]]+=1
        for z in label_tags:
            trues[z[0]]+=1
        for z in list(set(label_tags)&set(predicted_tags)):
            true_positive[z[0]]+=1
    avg_recall = 0.0
    avg_precision =0.0
    num_candidates=0


    #print positives,trues,true_positive
    for l in labels:
        if trues[l] ==0:
            recall =0
        else:
            recall=float(true_positive[l])/float(trues[l])
        if positives[l] ==0:
            precision =0
        else:
            precision=float(true_positive[l])/float(positives[l])
        if (recall+precision) ==0:
            f1 =0
        else:
            f1=2.0*recall*precision/(recall+precision)
        if l != IGNORE_TAG:
            avg_recall +=float(trues[l])*float(recall)
            avg_precision+=float(trues[l])*float(precision)
            num_candidates+=trues[l]
        if verbose:
            print("The tag \'{0}\' has {1} elements and recall,precision,f1 ={2},{3}, {4}".format(l,trues[l],recall,precision,f1))
    if num_candidates >0:
        avg_recall =float(avg_recall)/float(num_candidates)
        avg_precision =float(avg_precision)/float(num_candidates)
    avg_f1 =0.0
    if (avg_recall+avg_precision) >0:
        avg_f1=2.0*float(avg_precision)*float(avg_recall)/(float(avg_recall)+float(avg_precision))
    if verbose:
        print("All medical tags collectively have {0} elements and recall,precision,f1 ={1},{2}, {3}".format(num_candidates,avg_recall,avg_precision,avg_f1))
    return avg_f1

def evaluator(l,p,metric_func=get_Exact_Metrics):
    metric_func(l,p)

def strip_BIO(l,p):
    for i,sent in enumerate(l):
        l[i]=[token[2:] if token[:2]=='B-' else token for token in l[i]]
    for i,sent in enumerate(p):
        p[i]=[token[2:] if token[:2]=='B-' else token for token in p[i]]
    return l,p


