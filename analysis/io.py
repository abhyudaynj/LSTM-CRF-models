import bionlp.utils.monitoring as mo
import pickle
import os
import numpy as np
from time import gmtime, strftime
import lasagne
import logging


sl = logging.getLogger(__name__)
SOURCE_DIR = "data/logs"


def load_monitoring(filename="monitor_2017-06-"):
    accuracy = {mo.TYPE_VALIDATION: [], mo.TYPE_TRAINING: []}
    loss = {mo.TYPE_VALIDATION: [], mo.TYPE_TRAINING: []}
    for file in os.listdir(SOURCE_DIR):
        if file.startswith(filename) and file.endswith('.pkl'):
            pkl = pickle.load(open(os.path.join(SOURCE_DIR, file), 'rb'))
            accuracy[mo.TYPE_VALIDATION].append(pkl[mo.TYPE_VALIDATION][mo.METRIC_ACC])
            accuracy[mo.TYPE_TRAINING].append(pkl[mo.TYPE_TRAINING][mo.METRIC_ACC])

            loss[mo.TYPE_VALIDATION].append(pkl[mo.TYPE_VALIDATION][mo.METRIC_LOSS_TOT])
            loss[mo.TYPE_TRAINING].append(pkl[mo.TYPE_TRAINING][mo.METRIC_LOSS_TOT])

    return {mo.METRIC_ACC: accuracy, mo.METRIC_LOSS_TOT: loss}


def parse_eval_line(line=""):
    fulltag, rest = line.split(' tag ')[1].split(' has ')
    tag = fulltag.replace("'", "")

    count, rest = rest.split(" elements ")
    recall, precision, f1 = rest.split("=")[1].split(",")

    return tag, {"count": int(count), "recall": float(recall), "precision": float(precision), "f1": float(f1)}


def load_eval_txt_file(filename="eval_2017-06-"):
    result = {}
    for file in os.listdir(SOURCE_DIR):
        if file.startswith(filename) and file.endswith(".txt"):
            with open(os.path.join(SOURCE_DIR, file), "r") as fh:
                for line in fh:
                    if line.startswith("The tag"):
                        tag, line_result = parse_eval_line(line)

                        if tag not in result:
                            result[tag] = {"count": [], "recall": [], "precision": [], "f1": []}
                        [result[tag][key].append(value) for key, value in line_result.items()]
    return result


def pickle_confusion_matrix(confusion_matrix, path):
    datetime = strftime("%Y-%m-%d_%H-%M-%S", gmtime())
    name, ext = os.path.splitext(path)
    path_with_time = '{0}_{1}{2}'.format(name, datetime, ext)
    with open(path_with_time, "wb") as cm_f:
        pickle.dump(confusion_matrix, cm_f)


def load_confusion_matrix(filename):
    with open(os.path.join(SOURCE_DIR, filename), "rb") as cm_f:
        return pickle.load(cm_f)


def read_confusion_matrix_values(confusion_matrix):
    tags = confusion_matrix._values
    matrix = np.array(confusion_matrix._confusion)
    return tags, matrix


def save_net_params_if_necessary(netd, params, w2i, t2i, umls_v):
    if 'final_layers' in netd and params['model'] is not 'None' and params['trainable'] is True:
        nn_values = lasagne.layers.get_all_param_values(netd['final_layers'])
        sl.info('Saving NN param values to {0}'.format(params['model']))
        relevant_params = dict(params)
        del relevant_params['dependency']
        nn_packet = {'params': relevant_params, 'nn': nn_values,
                     'w2i': w2i, 't2i': t2i, 'umls_vocab': umls_v}
        pickle.dump(nn_packet, open(params['model'], 'wb'))


def store_response(o, l, p, params, filename='response.pkl'):
    sl.info("Storing responses in {0}".format(filename))
    pickle.dump((params, o, l, p), open(filename, 'wb'))


def update_monitoring(data, filename='monitor.pkl'):
    pickle.dump(data, open(filename, 'wb'))
