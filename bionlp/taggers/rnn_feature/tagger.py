import numpy as np
from tqdm import tqdm
import pickle
import bionlp.evaluate.evaluation as evaluation
import json
import bionlp.utils.data_utils as data_utils
import bionlp.utils.monitoring as monitor
from bionlp.taggers.rnn_feature.networks.utils import get_crf_model
from analysis.io import save_net_params_if_necessary, update_monitoring, store_response
import errno
import logging
import lasagne
from . import tagger_utils as preprocess
from sklearn.utils import shuffle as sk_shuffle
import datetime
import os


np.random.seed(1337)  # for reproducibility
params = {}
X = U = Y = Z = Mask = i2t = t2i = w2i = i2w = splits = numTags = emb_w = []

sl = logging.getLogger(__name__)


# TODO: worker is not used
def train_NN(train, crf_output, lstm_output, train_indices, compute_cost,
             compute_acc, compute_cost_regularization, worker, netd):
    time_of_last_save = datetime.datetime.now()
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
        sl.error("Aborting training. Both 'epochs' and 'patience' are set to 0, "
                 "so the training would have no termination condition.")
    else:
        # train until the desired number of epochs is reached or until the patience runs out
        iter_num = 1
        while True:
            perform_training_iteration(train, compute_cost, compute_acc, compute_cost_regularization, x_train,
                                       mask_train, y_train, params, iter_num, num_batches)
            if iter_num >= params['epochs'] > 0:
                sl.info("Stopping because the final epoch (epoch {0}) has been reached".format(params['epochs']))
                break
            elif params['patience'] != 0:
                patience_has_ended, vals = \
                    check_for_patience(lstm_output, compute_cost, compute_acc, x_dev, mask_dev, y_dev, params, vals)
                if patience_has_ended:
                    sl.info("Stopping because my patience has reached its limit.")
                    break
            if params['model'] is not 'None' and params['save-interval-mins'] is not 0:
                time_since_last_save = datetime.datetime.now() - time_of_last_save
                mins_since_last_save = time_since_last_save.total_seconds() / 60.
                sl.debug("Minutes since last save: %f" % mins_since_last_save)
                if mins_since_last_save >= params['save-interval-mins']:
                    sl.info("Enough time has passed; save the network parameters")
                    save_net_params_if_necessary(netd, params, w2i, t2i, umls_v)
                    time_of_last_save = datetime.datetime.now()
            iter_num += 1
            if params['monitoring-file'] != 'None':
                update_monitoring(monitor.get_data(), params['monitoring-file'])

    sl.info("Final Validation eval")
    evaluation.evaluate_neuralnet(lstm_output, x_dev, mask_dev, y_dev, i2t, i2w, params,
                                  strict=True, is_final_eval=True)


def perform_training_iteration(train, compute_cost, compute_acc, compute_cost_regularization, x_train, mask_train,
                               y_train, params, iter_num, num_batches):
    try:
        iter_cost = 0.0
        iter_acc = 0.0
        iter_cost_regularization = 0.0
        sl.info(('Iteration number : {0}'.format(iter_num)))
        for x_i, m_i, y_i in tqdm(preprocess.iterate_minibatches(x_train, mask_train, y_train, params['batch-size']),
                                  total=num_batches, leave=False):
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
                                       monitor.METRIC_ACC: acc,
                                       monitor.METRIC_LOSS_NET_CRF: loss_net_crf,
                                       monitor.METRIC_LOSS_CRF: loss_crf,
                                       monitor.METRIC_LOSS_TOT: loss_tot
                                   })
        sl.info(('TRAINING : Accuracy = {0}'.format(acc)))
        sl.info(('TRAINING : Network+CRF loss = {0} CRF-regularization loss = {1} Total loss = {2}'.format(
            loss_net_crf, loss_crf, loss_tot)))
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
            val_acc, _ = evaluation.evaluate_neuralnet(lstm_output, x_dev, mask_dev, y_dev, i2t, i2w, params,
                                                       strict=True, verbose=False)
        else:
            val_acc, val_loss = callback_NN(compute_cost, compute_acc, x_dev, mask_dev, y_dev)
        vals.append(val_acc)
        vals = vals[1:]
        max_in = np.argmax(vals)
        sl.info("val acc argmax {1} : list is : {0}".format(vals, max_in))
        monitor.add_iteration_data(
            monitor.TYPE_VALIDATION, {monitor.METRIC_ACC: val_acc, monitor.METRIC_LOSS_TOT: val_loss}
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
    for indx, (x_i, m_i, y_i) in enumerate(preprocess.iterate_minibatches(
            X_test, mask_test, y_test, params['batch-size'])):
        val_loss += compute_cost(x_i[:, :, :1].astype('int32'), x_i[:, :, 1:].astype(
            'float32'), y_i.astype('float32'), m_i.astype('float32'))
        val_acc += compute_acc(x_i[:, :, :1].astype('int32'), x_i[:, :, 1:].astype(
            'float32'), y_i.astype('float32'), m_i.astype('float32'))
    sl.info(('VALIDATION : acc = {0} loss = {1}'.format(
        val_acc / num_valid_batches, val_loss / num_valid_batches)))
    return val_acc / num_valid_batches, val_loss / num_valid_batches


def driver(worker, xxx_todo_changeme):
    (train_i, test_i) = xxx_todo_changeme
    setup_nn = get_crf_model(params)
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

    netd = setup_nn(0, X[0:params['batch-size']], U[0:params['batch-size']],
                    Mask[0:params['batch-size']], Y[0:params['batch-size']], params, numTags, emb_w)
    crf_output, lstm_output, train, compute_cost, compute_acc, compute_cost_regularization = netd['crf_output'], netd[
        'lstm_output'], netd['train'], netd['compute_cost'], netd['compute_acc'], netd['compute_cost_regularization']

    if params['model']:
        net_params_filename = params['model']
        if os.path.isfile(net_params_filename):
            sl.info('Loading Network weights from {0}'.format(net_params_filename))
            nn_v_d = pickle.load(open(params['model'], 'rb'), encoding='latin1')
            lasagne.layers.set_all_param_values(netd['final_layers'], nn_v_d['nn'])
        else:
            sl.info("Can't load Network weights from {0}; continuing with random weights".format(net_params_filename))
    else:
        sl.info('No file to load Network weights from has been given')

    if 'trainable' in params and params['trainable'] is True:
        train_NN(train, crf_output, lstm_output, train_i, compute_cost,
                 compute_acc, compute_cost_regularization, worker, netd)

    save_net_params_if_necessary(netd, params, w2i, t2i, umls_v)

    if params['deploy'] == 1:
        _, results = evaluation.evaluate_neuralnet(
            lstm_output, np.concatenate([X[test_i].astype('float32'), U[test_i]], axis=2),
            Mask[test_i], Y[test_i], i2t, i2w, params, z_test=Z[test_i], strict=True, verbose=False)
    else:
        sl.info("Final evalution for this fold on testing set")
        _, results = evaluation.evaluate_neuralnet(
            lstm_output, np.concatenate([X[test_i].astype('float32'), U[test_i]], axis=2),
            Mask[test_i], Y[test_i], i2t, i2w, params, z_test=Z[test_i], strict=True, verbose=True)
    return results


def single_run():
    worker = 0
    o, l, p = driver(worker, splits[worker])
    if params['error-analysis'] != 'None':
        store_response(o, l, p, params, filename=params['error-analysis'])
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
    evaluation.get_Approx_Metrics(
        flat_label, flat_predicted, pre_msg='NN_VALIDATION:', flat_list=True)
    sl.info("STRICT ---")
    evaluation.get_Exact_Metrics(label_sent, predicted_sent)
    if params['error-analysis'] != 'None':
        store_response(original_sent, label_sent, predicted_sent, params, filename=params['error-analysis'])
    return original_sent, label_sent, predicted_sent


def deploy_run(splits, params):
    sl.info('Running in Deploy Mode. This means I will train on all available data. '
            'Final Eval metrics will not be meaningful')
    if params['model'] is 'None':
        sl.warning(
            'No model file location is provided. Without a model file location, '
            'the deployable model will not be saved anywhere')
    res_dict = driver(0, (np.array(list(range(len(Y)))), splits[1]))
    return res_dict


def evaluate_run():
    sl.info('Running in Evaluate Mode. I will not learn anything, '
            'only evaluate the existing model on the entire provided data ')
    o, l, p = driver(0, (np.array(list(range(len(Y)))),
                         np.array(list(range(len(Y))))))
    if params['error-analysis'] != 'None':
        store_response(o, l, p, params, filename=params['error-analysis'])
    return o, l, p


def rnn_train(dataset, config_params, vocab, umls_vocab):
    global params

    global X, U, Y, Z, Mask, i2t, t2i, w2i, i2w, splits, numTags, emb_w, umls_v
    params = config_params
    umls_v = umls_vocab

    # Preparing Dataset
    sl.info('Preparing entire dataset for Neural Net computation ...')
    (X, U, Z, Y), numTags, emb_w, t2i, w2i = preprocess.load_data(
        dataset, params, entire_note=params['document'], vocab=vocab)

    X, U, Y, Z, Mask = preprocess.pad_and_mask(X, U, Y, Z, params['maxlen'])
    sl.info('Total non zero entries in the Mask Inputs are {0}. This number should be equal to '
            'the total number of tokens in the entire dataset'.format(sum(sum(_) for _ in Mask)))
    if params['shuffle'] == 1:
        X, U, Y, Z, Mask = sk_shuffle(X, U, Y, Z, Mask, random_state=0)
    i2t = {v: k for k, v in list(t2i.items())}
    i2w = {v: k for k, v in list(w2i.items())}
    splits = data_utils.make_cross_validation_sets(
        len(Y), params['folds'], training_percent=params['training-percent'])

    o = l = p = None
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
    return o, l, p
