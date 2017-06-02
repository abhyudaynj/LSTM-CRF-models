import numpy as np
from tqdm import tqdm
import pickle
import argparse
import bionlp.evaluate.evaluation as eval_metrics
import sys
import random
import json
np.random.seed(1337)  # for reproducibility
import bionlp.utils.data_utils as data_utils
import bionlp.utils.monitoring as monitor
import errno
from nltk.metrics import ConfusionMatrix
import itertools
import time
import logging
import lasagne
from . import tagger_utils as preprocess
from sklearn.metrics import f1_score
from sklearn.utils import shuffle as sk_shuffle
import theano.tensor as T
import theano

params = {}
X = U = Y = Z = Mask = i2t = t2i = w2i = i2w = splits = numTags = emb_w = []
setup_NN = {}

sl = logging.getLogger(__name__)


# TODO: worker is not used
def train_NN(train, crf_output, lstm_output, train_indices, compute_cost, compute_acc, compute_cost_regularization, worker):
    if params['patience'] != 0:
        vals = [0.0] * params['patience']
    sl.info('Dividing the training set into {0} % training and {1} % dev set'.format(
        100 - params['dev'], params['dev']))
    train_i = np.copy(train_indices)
    np.random.shuffle(train_i)
    dev_length = len(train_indices) * params['dev'] // 100
    dev_i = train_i[:dev_length]
    train_i = train_i[dev_length:]
    sl.info('{0} training set samples and {1} dev set samples'.format(
        len(train_i), len(dev_i)))
    x_train = np.concatenate(
        [X[train_i].astype('float32'), U[train_i]], axis=2)
    sl.info('isnan is {0}'.format(np.isnan(np.sum(x_train))))
    y_train = Y[train_i]

    mask_train = Mask[train_i]

    x_dev = np.concatenate([X[dev_i].astype('float32'), U[dev_i]], axis=2)
    y_dev = Y[dev_i]
    mask_dev = Mask[dev_i]
    num_batches = float(sum(1 for _ in preprocess.iterate_minibatches(
        x_train, mask_train, y_train, params['batch-size'])))

    # perform training until the defined termination condition has been reached
    if params['epochs'] == 0 and params['patience'] == 0:
        # if epochs are 0 and patience is 0, the training can't end, so don't start it
        sl.error("Aborting training. Both 'epochs' and 'patience' are set to 0, so the training would have no termination condition.")
    else:
        # train until the desired number of epochs is reached or until the patience runs out
        iter_num = 1
        while True:
            perform_training_iteration(train, compute_cost, compute_acc, compute_cost_regularization, x_train, mask_train, y_train, params, iter_num, num_batches)
            if params['epochs'] > 0 and iter_num >= params['epochs']:
                sl.info("Stopping because the final epoch (epoch {0}) has been reached".format(params['epochs']))
                break
            elif params['patience'] != 0:
                patience_has_ended, vals = check_for_patience(lstm_output, compute_cost, compute_acc, x_dev, mask_dev, y_dev, params, vals)
                if patience_has_ended:
                    sl.info("Stopping because my patience has reached its limit.")
                    break
            
            iter_num += 1
            if params['monitoring-file'] != 'None':
                update_monitoring(params['monitoring-file'])

    sl.info("Final Validation eval")
    evaluate_neuralnet(lstm_output, x_dev, mask_dev, y_dev, strict=True)


def perform_training_iteration(train, compute_cost, compute_acc, compute_cost_regularization, x_train, mask_train, y_train, params, iter_num, num_batches):
    try:
        iter_cost = 0.0
        iter_acc = 0.0
        iter_cost_regularization = 0.0
        sl.info(('Iteration number : {0}'.format(iter_num)))
        for x_i, m_i, y_i in tqdm(preprocess.iterate_minibatches(x_train, mask_train, y_train, params['batch-size']), total=num_batches, leave=False):
            train(x_i[:, :, :1].astype('int32'), x_i[:, :, 1:].astype(
                'float32'), y_i.astype('float32'), m_i.astype('float32'))
            iter_cost += compute_cost(x_i[:, :, :1].astype('int32'), x_i[:, :, 1:].astype(
                'float32'), y_i.astype('float32'), m_i.astype('float32'))
            iter_acc += compute_acc(x_i[:, :, :1].astype('int32'), x_i[:, :, 1:].astype(
                'float32'), y_i.astype('float32'), m_i.astype('float32'))
            iter_cost_regularization += compute_cost_regularization()

        acc, loss_net_crf, loss_crf, loss_tot = \
            np.array([iter_acc, iter_cost, iter_cost_regularization, iter_cost + iter_cost_regularization])/num_batches
        monitor.add_iteration_data(monitor.TYPE_TRAINING,
                                   {
                                       monitor.KEY_ACC: acc,
                                       monitor.KEY_LOSS_NET_CRF: loss_net_crf,
                                       monitor.KEY_LOSS_CRF: loss_crf,
                                       monitor.KEY_LOSS_TOT: loss_tot
                                   })
        sl.info(('TRAINING : Accuracy = {0}'.format(acc)))
        sl.info(('TRAINING : Network+CRF loss = {0} CRF-regularization loss = {1} Total loss = {2}'.format(
            loss_net_crf, loss_crf, loss_tot)))
        # if iter_num % 5 == 0:
            # TODO: res is currently not used
            # res = evaluate_neuralnet(lstm_output, x_dev, mask_dev, y_dev, strict=True)
    except IOError as e:
        if e.errno != errno.EINTR:
            raise
        else:
            sl.error(" EINTR ERROR CAUGHT. YET AGAIN ")


def check_for_patience(lstm_output, compute_cost, compute_acc, x_dev, mask_dev, y_dev, params, vals):
    patience_has_ended = False
    try:
        if params['patience-mode'] != 0:
            val_loss = 0
            val_acc, _ = evaluate_neuralnet(lstm_output, x_dev, mask_dev, y_dev, strict=True, verbose=False)
        else:
            val_acc, val_loss = callback_NN(compute_cost, compute_acc, x_dev, mask_dev, y_dev)
        vals.append(val_acc)
        vals = vals[1:]
        max_in = np.argmax(vals)
        sl.info("val acc argmax {1} : list is : {0}".format(vals, max_in))
        monitor.add_iteration_data(
            monitor.TYPE_VALIDATION, {monitor.KEY_ACC: val_acc, monitor.KEY_LOSS_TOT: val_loss}
        )
        if max_in == 0:
            patience_has_ended = True
    except IOError as e:
        if e.errno != errno.EINTR:
            raise
        else:
            sl.info(" EINTR ERROR CAUGHT. YET AGAIN ")
    return patience_has_ended, vals


def callback_NN(compute_cost, compute_acc, X_test, mask_test, y_test):
    num_valid_batches = float(sum(1 for _ in preprocess.iterate_minibatches(
        X_test, mask_test, y_test, params['batch-size'])))
    sl.info(('num_valid_batches {0}'.format(num_valid_batches)))
    sl.info('Executing validation Callback')
    val_loss = 0.0
    val_acc = 0.0
    #num_batches=float(sum(1 for _ in preprocess.iterate_minibatches(X_test,mask_test,y_test,params['batch-size'])))
    for indx, (x_i, m_i, y_i) in enumerate(preprocess.iterate_minibatches(X_test, mask_test, y_test, params['batch-size'])):
        val_loss += compute_cost(x_i[:, :, :1].astype('int32'), x_i[:, :, 1:].astype(
            'float32'), y_i.astype('float32'), m_i.astype('float32'))
        val_acc += compute_acc(x_i[:, :, :1].astype('int32'), x_i[:, :, 1:].astype(
            'float32'), y_i.astype('float32'), m_i.astype('float32'))
    sl.info(('VALIDATION : acc = {0} loss = {1}'.format(
        val_acc / num_valid_batches, val_loss / num_valid_batches)))
    return val_acc / num_valid_batches, val_loss / num_valid_batches


def evaluate_neuralnet(lstm_output, X_test, mask_test, y_test, z_test=None, strict=False, verbose=True):
    if params['trainable'] is False and params['noeval'] == True:
        verbose = False
    if z_test == None:
        sl.info('z_test not provided. Using mask vector as a placeholder')
        z_test = mask_test
    sl.info(('Mask len test', len(mask_test)))
    predicted = []
    predicted_sent = []
    label = []
    label_sent = []
    original_sent = []
    for indx, (x_i, m_i, y_i, z_i) in enumerate(preprocess.iterate_minibatches(X_test, mask_test, y_test, params['batch-size'], z_test)):
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
    res = eval_metrics.get_Approx_Metrics([i2t[l] for l in label], [
                                          i2t[l] for l in predicted], verbose=verbose, preMsg='NN:', flat_list=True)
    if strict:
        res = eval_metrics.get_Exact_Metrics(
            label_sent, predicted_sent, verbose=verbose)
        sl.info('Output number of tokens are {0}'.format(
            sum(len(_) for _ in predicted_sent)))

    return res, (original_sent, label_sent, predicted_sent)


def driver(worker, xxx_todo_changeme):
    (train_i, test_i) = xxx_todo_changeme
    if worker == 0:
        sl.info('Embedding Shape : {0}'.format(emb_w.shape))
        sl.info('{0} train sequences'.format(len(X[train_i])))
        sl.info('{0} test sequences'.format(len(X[test_i])))
        sl.info('Number of tags {0}'.format(numTags))
        sl.info('X train Sanity check: {0}'.format(
            np.amax(np.amax(X[train_i]))))
        sl.info('X test Sanity check :{0}'.format(np.amax(np.amax(X[test_i]))))
        sl.info('X_train shape:{0}'.format(X[train_i].shape))
        sl.info('X_test shape:{0}'.format(X[test_i].shape))

        sl.info('Z_train shape:{0}'.format(Z[train_i].shape))
        sl.info('Z_test shape:{0}'.format(Z[test_i].shape))

        sl.info('U_train shape:{0}'.format(U[train_i].shape))
        sl.info('U_test shape:{0}'.format(U[test_i].shape))

        sl.info('mask_train shape:{0}'.format(Mask[train_i].shape))
        sl.info('mask_test shape:{0}'.format(Mask[test_i].shape))

        sl.info('Y_train shape:{0}'.format(Y[train_i].shape))
        sl.info('Y_test shape:{0}'.format(Y[test_i].shape))

    netd = setup_NN(0, X[0:params['batch-size']], U[0:params['batch-size']],
                    Mask[0:params['batch-size']], Y[0:params['batch-size']], params, numTags, emb_w)
    crf_output, lstm_output, train, compute_cost, compute_acc, compute_cost_regularization = netd['crf_output'], netd[
        'lstm_output'], netd['train'], netd['compute_cost'], netd['compute_acc'], netd['compute_cost_regularization']
    if 'trainable' in params and params['trainable'] == True:
        train_NN(train, crf_output, lstm_output, train_i, compute_cost,
                 compute_acc, compute_cost_regularization, worker)
    else:
        sl.info('Trainable is off. Loading Network weights from {0}'.format(
            params['model']))
        nn_v_d = pickle.load(open(params['model'], 'rb'), encoding='latin1')
        lasagne.layers.set_all_param_values(netd['final_layers'], nn_v_d['nn'])

    if 'final_layers' in netd and params['model'] is not 'None' and params['trainable'] is True:
        nn_values = lasagne.layers.get_all_param_values(netd['final_layers'])
        sl.info('Saving NN param values to {0}'.format(params['model']))
        relevant_params = dict(params)
        del relevant_params['dependency']
        nn_packet = {'params': relevant_params, 'nn': nn_values,
                     'w2i': w2i, 't2i': t2i, 'umls_vocab': umls_v}
        pickle.dump(nn_packet, open(params['model'], 'wb'))

    if params['deploy'] == 1:
        _, results = evaluate_neuralnet(lstm_output, np.concatenate([X[test_i].astype(
            'float32'), U[test_i]], axis=2), Mask[test_i], Y[test_i], Z[test_i], strict=True, verbose=False)
    else:
        sl.info("Final evalution for this fold on testing set")
        _, results = evaluate_neuralnet(lstm_output, np.concatenate([X[test_i].astype(
            'float32'), U[test_i]], axis=2), Mask[test_i], Y[test_i], Z[test_i], strict=True, verbose=True)
    return results


def store_response(o, l, p, filename='response.pkl'):
    sl.info("Storing responses in {0}".format(filename))
    pickle.dump((params, o, l, p), open(filename, 'wb'))


def update_monitoring(filename='monitor.pkl'):
    pickle.dump(monitor.get_data(), open(filename, 'wb'))


def single_run():
    worker = 0
    o, l, p = driver(worker, splits[worker])
    if params['error-analysis'] != 'None':
        store_response(o, l, p, params['error-analysis'])
    return o, l, p


def cross_validation_run():
    label_sent = []
    predicted_sent = []
    original_sent = []
    for worker in range(len(splits)):
        sl.info("########### Cross Validation run : {0}".format(worker))
        o, l, p = driver(worker, splits[worker])
        label_sent += l
        predicted_sent += p
        original_sent += o
    sl.info("#######################VALIDATED SET ########")
    flat_label = [word for sentenc in label_sent for word in sentenc]
    flat_predicted = [word for sentenc in predicted_sent for word in sentenc]
    eval_metrics.get_Approx_Metrics(
        flat_label, flat_predicted, preMsg='NN_VALIDATION:', flat_list=True)
    sl.info("STRICT ---")
    eval_metrics.get_Exact_Metrics(label_sent, predicted_sent)
    if params['error-analysis'] != 'None':
        store_response(original_sent, label_sent,
                       predicted_sent, params['error-analysis'])
    return original_sent, label_sent, predicted_sent


def deploy_run(splits, params):
    sl.info('Running in Deploy Mode. This means I will train on all available data. Final Eval metrics will not be meaningful')
    if params['model'] is 'None':
        sl.warning(
            'No model file location is provided. Without a model file location, the deployable model will not be saved anywhere')
    res_dict = driver(0, (np.array(list(range(len(Y)))), splits[1]))
    return res_dict


def evaluate_run():
    sl.info('Running in Evaluate Mode. I will not learn anything, only evaluate the existing model on the entire provided data ')
    o, l, p = driver(0, (np.array(list(range(len(Y)))),
                         np.array(list(range(len(Y))))))
    if params['error-analysis'] != 'None':
        store_response(o, l, p, params['error-analysis'])
    return o, l, p


def rnn_train(dataset, config_params, vocab, umls_vocab):
    global params, setup_NN

    global X, U, Y, Z, Mask, i2t, t2i, w2i, i2w, splits, numTags, emb_w, umls_v
    params = config_params
    umls_v = umls_vocab
    if 'CRF_MODEL_ON' in params and params['CRF_MODEL_ON']:
        sl.info('CRF IS ON. CRF_MODELS WILL BE USED')
        if params['mode'] == 1:
            from bionlp.taggers.rnn_feature.networks.approx_network import setup_NN
            sl.info('MODE :Using the Approximate Message Passing framework')
        elif params['mode'] == -1:
            from bionlp.taggers.rnn_feature.networks.network import setup_NN
            sl.info('MODE : Modeling only the unary potentials')
        else:
            sl.info('MODE : Modeling both unary and binary potentials')
            from bionlp.taggers.rnn_feature.networks.dual_network import setup_NN
    else:
        sl.info(
            'CRF IS NOT ON. This tagger only supports CRF models. A default CRF_MODEL will be used.')
        params['mode'] = 1
        params['CRF_MODEL_ON'] = True
        from bionlp.taggers.rnn_feature.networks.approx_network import setup_NN
        sl.info('MODE :Using the Approximate Message Passing framework')

    sl.info('Using the parameters:\n {0}'.format(json.dumps(params, indent=2)))

    # Preparing Dataset

    sl.info('Preparing entire dataset for Neural Net computation ...')
    (X, U, Z, Y), numTags, emb_w, t2i, w2i = preprocess.load_data(
        dataset, params, entire_note=params['document'], vocab=vocab)

    X, U, Y, Z, Mask = preprocess.pad_and_mask(X, U, Y, Z, params['maxlen'])
    sl.info('Total non zero entries in the Mask Inputs are {0}. This number should be equal to total number of tokens in the entire dataset'.format(
        sum(sum(_) for _ in Mask)))
    if params['shuffle'] == 1:
        X, U, Y, Z, Mask = sk_shuffle(X, U, Y, Z, Mask, random_state=0)
    i2t = {v: k for k, v in list(t2i.items())}
    i2w = {v: k for k, v in list(w2i.items())}
    splits = data_utils.make_cross_validation_sets(
        len(Y), params['folds'], training_percent=params['training-percent'])
    try:
        if params['trainable'] is False:
            (o, l, p) = evaluate_run()
        elif params['deploy'] == 1:
            (o, l, p) = deploy_run(splits[0], params)
        elif params['cross-validation'] == 0:
            (o, l, p) = single_run()
        else:
            (o, l, p) = cross_validation_run()
    except IOError as e:
        if e.errno != errno.EINTR:
            raise
        else:
            sl.error(" EINTR ERROR CAUGHT. YET AGAIN ")
    sl.info('Using the parameters:\n {0}'.format(json.dumps(params, indent=2)))
    return (o, l, p)
