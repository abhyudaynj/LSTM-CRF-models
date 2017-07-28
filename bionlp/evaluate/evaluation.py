from bionlp.taggers.rnn_feature.tagger_utils import iterate_minibatches
import analysis.io as io
import numpy as np
import itertools
import logging
from nltk.metrics import ConfusionMatrix
from sklearn.metrics import f1_score, recall_score, precision_score
import collections


IGNORE_TAG = 'None'
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def get_labels(label, predicted):
    labels = list(set(itertools.chain.from_iterable(label)) |
                  set(itertools.chain.from_iterable(predicted)))
    return labels


def get_Approx_Metrics(y_true, y_pred, verbose=True, pre_msg='', flat_list=False):
    if verbose:
        print('------------------------ Approx Metrics---------------------------')
    if flat_list:
        z_true = y_true
        z_pred = y_pred
    else:
        z_true = list(itertools.chain.from_iterable(y_true))
        z_pred = list(itertools.chain.from_iterable(y_pred))
    z_true = [token[2:] if token[:2] == 'B-' else token for token in z_true]
    z_pred = [token[2:] if token[:2] == 'B-' else token for token in z_pred]
    label_dict = {x: i for i, x in enumerate(list(set(z_true) | set(z_pred)))}
    freq_dict = collections.Counter(z_true)
    z_true = [label_dict[x] for x in z_true]
    z_pred = [label_dict[x] for x in z_pred]
    f1s = f1_score(z_true, z_pred, average=None)
    rs = recall_score(z_true, z_pred, average=None)
    ps = precision_score(z_true, z_pred, average=None)
    f1_none = []
    avg_recall = avg_precision = avg_f1 = 0.0
    for i in label_dict:
        if verbose:
            print(("{5} The tag \'{0}\' has {1} elements and recall,precision,f1 ={3},{4}, {2}".format(
                i, freq_dict[i], f1s[label_dict[i]], rs[label_dict[i]], ps[label_dict[i]], pre_msg)))
        if i != 'None' and i != '|O':
            f1_none = f1_none + [(f1s[label_dict[i]], freq_dict[i]), ]
            avg_recall += float(rs[label_dict[i]]) * float(freq_dict[i])
            avg_precision += float(ps[label_dict[i]]) * float(freq_dict[i])
    intermediate_sum = sum([float(z[1]) for z in f1_none])
    if intermediate_sum == 0:
        intermediate_sum += 1
    avg_recall = float(avg_recall) / float(intermediate_sum)
    avg_precision = float(avg_precision) / float(intermediate_sum)
    if (float(avg_recall) + float(avg_precision)) != 0.0:
        avg_f1 = 2.0 * float(avg_precision) * float(avg_recall) /  (float(avg_recall) + float(avg_precision))
    if verbose:
        print(("All medical tags collectively have {0} elements and recall,precision,f1 ={1},{2}, {3}".format(
            intermediate_sum, avg_recall, avg_precision, avg_f1)))
    return avg_f1


def get_Exact_Metrics(true, predicted, verbose=True, is_final_eval=False, final_eval_out_file='None'):
    true, predicted = strip_bio(true, predicted)
    if verbose:
        print('------------------------ Exact Metrics---------------------------')
        cm = get_confusion_matrix(true, predicted)
        if is_final_eval and final_eval_out_file is not 'None':
            io.pickle_confusion_matrix(cm, final_eval_out_file)
    labels = get_labels(true, predicted)
    true_positive = {label: 0 for label in labels}
    trues = {label: 0 for label in labels}
    positives = {label: 0 for label in labels}
    for i, sent in enumerate(true):
        if sent.__len__() == 0:
            continue
        label_tags = []
        predicted_tags = []
        j = 0
        tag = 'Nothing'
        pos = []
        while j < len(sent):
            if tag != sent[j]:
                if tag != 'Nothing':
                    label_tags.append((tag, tuple(pos)))
                pos = []
                pos.append(j)
                tag = sent[j]
            else:
                pos.append(j)
            j += 1
        label_tags.append((tag, tuple(pos)))

        j = 0
        tag = 'Nothing'
        pos = []
        psent = predicted[i]
        while j < len(psent):
            if tag != psent[j]:
                if tag != 'Nothing':
                    predicted_tags.append((tag, tuple(pos)))
                pos = []
                pos.append(j)
                tag = psent[j]
            else:
                pos.append(j)
            j += 1
        predicted_tags.append((tag, tuple(pos)))
        for z in predicted_tags:
            positives[z[0]] += 1
        for z in label_tags:
            trues[z[0]] += 1
        for z in list(set(label_tags) & set(predicted_tags)):
            true_positive[z[0]] += 1
    avg_recall = 0.0
    avg_precision = 0.0
    num_candidates = 0

    # print positives,trues,true_positive
    for l in labels:
        if trues[l] == 0:
            recall = 0
        else:
            recall = float(true_positive[l]) / float(trues[l])
        if positives[l] == 0:
            precision = 0
        else:
            precision = float(true_positive[l]) / float(positives[l])
        if (recall + precision) == 0:
            f1 = 0
        else:
            f1 = 2.0 * recall * precision / (recall + precision)
        if l != IGNORE_TAG:
            avg_recall += float(trues[l]) * float(recall)
            avg_precision += float(trues[l]) * float(precision)
            num_candidates += trues[l]
        if verbose:
            msg = "The tag \'{0}\' has {1} elements and recall,precision,f1 ={2},{3}, {4}".format(
                l, trues[l], recall, precision, f1)
            print(msg)

    if num_candidates > 0:
        avg_recall = float(avg_recall) / float(num_candidates)
        avg_precision = float(avg_precision) / float(num_candidates)
    avg_f1 = 0.0
    if (avg_recall + avg_precision) > 0:
        avg_f1 = 2.0 * float(avg_precision) * float(avg_recall) / \
            (float(avg_recall) + float(avg_precision))
    if verbose:
        msg = "All medical tags collectively have {0} elements and recall,precision,f1 ={1},{2}, {3}".format(
            num_candidates, avg_recall, avg_precision, avg_f1)
        print(msg)

    return avg_f1


def evaluate_neuralnet(lstm_output, X_test, mask_test, y_test, i2t, i2w, params,
                       z_test=None, strict=False, verbose=True, is_final_eval=False):
    if params['trainable'] is False and params['noeval'] is True:
        verbose = False
    if z_test is None:
        logger.info('z_test not provided. Using mask vector as a placeholder')
        z_test = mask_test
    logger.info(('Mask len test', len(mask_test)))
    predicted = []
    predicted_sent = []
    label = []
    label_sent = []
    original_sent = []
    for indx, (x_i, m_i, y_i, z_i) in enumerate(
            iterate_minibatches(X_test, mask_test, y_test, params['batch-size'], z_test)):
        for sent_ind, m_ind in enumerate(m_i):
            o_sent = x_i[sent_ind][m_i[sent_ind] == 1].tolist()
            original_sent.append(([i2w[int(l[0])]
                                   for l in o_sent], z_i[sent_ind]))
        y_p = lstm_output(x_i[:, :, :1].astype(
            'int32'), x_i[:, :, 1:].astype('float32'), m_i.astype('float32'))
        for sent_ind, m_ind in enumerate(m_i):
            l_sent = np.argmax(
                y_i[sent_ind][m_i[sent_ind] == 1], axis=1).tolist()
            p_sent = np.argmax(
                y_p[sent_ind][m_i[sent_ind] == 1], axis=1).tolist()
            predicted_sent.append([i2t[l] for l in p_sent])
            label_sent.append([i2t[l] for l in l_sent])
        m_if = m_i.flatten()
        label += np.argmax(y_i, axis=2).flatten()[m_if == 1].tolist()
        predicted += np.argmax(y_p, axis=2).flatten()[m_if == 1].tolist()
    res = get_Approx_Metrics([i2t[l] for l in label],
                             [i2t[l] for l in predicted],
                             verbose=verbose, pre_msg='NN:', flat_list=True)
    if strict:
        res = get_Exact_Metrics(label_sent, predicted_sent, verbose=verbose, is_final_eval=is_final_eval,
                                final_eval_out_file=params['eval-file'])
        logger.info('Output number of tokens are {0}'.format(
            sum(len(_) for _ in predicted_sent)))

    return res, (original_sent, label_sent, predicted_sent)


def evaluator(l, p, metric_func=get_Exact_Metrics):
    metric_func(l, p)


def strip_bio(l, p):
    for i, sent in enumerate(l):
        l[i] = [token[2:] if token[:2] == 'B-' else token for token in l[i]]
    for i, sent in enumerate(p):
        p[i] = [token[2:] if token[:2] == 'B-' else token for token in p[i]]
    return l, p


def get_confusion_matrix(true, predicted):
    # Confusion Matrix is only valid for partial evaluation.
    true_chain = list(itertools.chain.from_iterable(true))
    predicted_chain = list(itertools.chain.from_iterable(predicted))
    cm = ConfusionMatrix(true_chain, predicted_chain)
    msg = "Confusion Matrix of combined folds (partial evaluation)\n{0}".format(cm)
    logger.info(msg)
    return cm
