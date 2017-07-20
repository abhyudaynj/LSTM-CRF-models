import numpy as np

from .crf_dual_layer import DualCRFLayer, get_crf_training_loss
import lasagne
import time
import theano
import logging
from .base_network import get_base_network
import sys
import theano.tensor as T
from bionlp.utils.utils import theano_logsumexp
logging.basicConfig()
logger = logging.getLogger(__name__)


def setup_NN(worker, x_in, u_in, mask_in, y_in, params, numTags, emb_w):
    premodel = time.time()
    logger.info('Loading CRF-RNN network format with pairwise modeling. '
                'Forward Backward will be used for calculating the marginals while training. '
                'The output sequence will be calculated using posterior decoding.')

    crf_layer, pairwise, l_in, l_mask, l_u_in = get_base_network(x_in, u_in, mask_in, y_in, params, numTags, emb_w)
    t_out = T.tensor3()

    mid_out = lasagne.layers.get_output(crf_layer, deterministic=True)
    mid_output = theano.function(
        [l_in.input_var, l_u_in.input_var, l_mask.input_var], mid_out)
    logger.info("output shape for for unary mid layer {0}".format(mid_output(
        x_in.astype('int32'), u_in.astype('float32'), mask_in.astype('float32')).shape))
    logger.info("output sum for for unary mid layer {0}".format(np.sum(mid_output(
        x_in.astype('int32'), u_in.astype('float32'), mask_in.astype('float32'))[0, :, :], axis=1)))

    sum_layer = DualCRFLayer([crf_layer, pairwise, l_mask], mask_input=True)

    outp = lasagne.layers.get_output(sum_layer, deterministic=False)
    eval_out = lasagne.layers.get_output(sum_layer, deterministic=True)

    crf_output = theano.function(
        [l_in.input_var, l_u_in.input_var, l_mask.input_var], eval_out)
    lstm_output = crf_output  # Included for future functionality
    print(("output shape for theano net", lstm_output(x_in.astype(
        'int32'), u_in.astype('float32'), mask_in.astype('float32')).shape))
    eval_cost = T.mean((eval_out - t_out)**2)

    all_params = lasagne.layers.get_all_params(sum_layer, trainable=True)
    logger.info('Params :{0}'.format(all_params))
    logger.info(
        '\'l1\' and \'l2\' Regularization is applied to all trainable parameters in the network')
    l2_cost = params['l2'] * lasagne.regularization.apply_penalty(
        all_params, lasagne.regularization.l2)
    l1_cost = params['l1'] * lasagne.regularization.apply_penalty(
        all_params, lasagne.regularization.l1)
    regularization_losses = l2_cost + l1_cost

    num_params = lasagne.layers.count_params(sum_layer, trainable=True)
    print(('Number of parameters: {0}'.format(num_params)))

    crf_train_loss = get_crf_training_loss(
        sum_layer, pairwise, t_out, numTags, params, x_in, u_in, y_in, mask_in, l_in, l_u_in, l_mask)
    cost = T.mean(crf_train_loss) + regularization_losses
    cost_loss = T.mean(crf_train_loss)
    cost_regularization = regularization_losses

    updates = lasagne.updates.adagrad(
        cost, all_params, learning_rate=params['learning-rate'])
    if params['momentum'] == 2:
        updates = lasagne.updates.apply_nesterov_momentum(
            updates, all_params, momentum=0.9)
        logger.info('Using Nesterov\'s momentum')
    if params['momentum'] == 1:
        updates = lasagne.updates.apply_momentum(
            updates, all_params, momentum=0.9)
        logger.info('Using Momentum')
    if params['momentum'] == 0:
        logger.warning('Not using any momentum')

    train = theano.function(
        [l_in.input_var, l_u_in.input_var, t_out, l_mask.input_var], cost, updates=updates)

    compute_cost = theano.function(
        [l_in.input_var, l_u_in.input_var, t_out, l_mask.input_var], cost)
    compute_cost_loss = theano.function(
        [l_in.input_var, l_u_in.input_var, t_out, l_mask.input_var], cost_loss)

    compute_cost_regularization = theano.function([], cost_regularization)
    acc_ = T.sum(T.eq(T.argmax(eval_out, axis=2), T.argmax(
        t_out, axis=2)) * l_mask.input_var) / T.sum(l_mask.input_var)
    compute_acc = theano.function(
        [l_in.input_var, l_u_in.input_var, t_out, l_mask.input_var], acc_)
    print(('Time to build and compile model {0}'.format(
        time.time() - premodel)))

    return {'crf_output': crf_output, 'lstm_output': lstm_output, 'train': train, 'compute_cost': compute_cost,
            'compute_acc': compute_acc, 'compute_cost_loss': compute_cost_loss,
            'compute_cost_regularization': compute_cost_regularization, 'final_layers': sum_layer}
