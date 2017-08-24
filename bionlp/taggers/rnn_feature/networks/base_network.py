import lasagne
import time
import theano
import logging
import sys
import theano.tensor as T
from bionlp.utils.utils import theano_logsumexp
logging.basicConfig()
logger = logging.getLogger(__name__)


def get_base_network(x_in, u_in, mask_in, y_in, params, numTags, emb_w):
    batch_size = x_in.shape[0]
    logger.info('X train batch size {0}'.format(x_in.shape))
    logger.info('U train batch size {0}'.format(u_in.shape))
    logger.info('Y train batch size {0}'.format(y_in.shape))
    logger.info('mask train batch size {0}'.format(mask_in.shape))
    xt = theano.shared(x_in)

    l_in = lasagne.layers.InputLayer(
        shape=(None, params['maxlen'], 1), input_var=xt.astype('int32'))
    l_u_in = lasagne.layers.InputLayer(
        shape=(None, params['maxlen'], u_in.shape[2]))
    l_mask = lasagne.layers.InputLayer(shape=(None, params['maxlen']))
    # ---------- Replaced -------------

    l_u_reshape = lasagne.layers.ReshapeLayer(l_u_in, (-1, u_in.shape[2]))
    l_e_out = lasagne.layers.get_output(l_u_reshape)
    l_o_f = theano.function([l_u_in.input_var], l_e_out)
    logger.info("output shape for reshaped feature net = {0}".format(
        l_o_f(u_in.astype('float32')).shape))
    logger.info('Using {0} dimensional layer for extra features'.format(
        params['feature1']))
    l_u_dense = lasagne.layers.DenseLayer(l_u_reshape, params['feature1'], name='Feature-embedding')
    l_u_proc = lasagne.layers.ReshapeLayer(
        l_u_dense, (batch_size, params['maxlen'], 1, params['feature1']))
    if params['word2vec'] == 1:
        l_emb = lasagne.layers.EmbeddingLayer(
            l_in, emb_w.shape[0], emb_w.shape[1], W=emb_w.astype('float32'), name='Word-embedding')
        l_emb.add_param(l_emb.W, l_emb.W.get_value().shape, trainable=params['emb1'])
    else:
        l_emb = lasagne.layers.EmbeddingLayer(
            l_in, emb_w.shape[0], emb_w.shape[1], name='Word embedding')
    if params['emb2'] > 0:
        l_emb1 = lasagne.layers.EmbeddingLayer(l_in, emb_w.shape[0], params['emb2'])
        l_emb = lasagne.layers.ConcatLayer([l_emb, l_emb1], axis=3, name='Word-embedding')

    l_e_out = lasagne.layers.get_output(l_emb)
    l_p_out = lasagne.layers.get_output(l_u_proc)
    l_o_f = theano.function([l_in.input_var], l_e_out)
    l_p_f = theano.function([l_u_in.input_var], l_p_out)
    logger.info("output shape for emb net ={0}".format(
        l_o_f(x_in.astype('int32')).shape))
    logger.info("output shape for feature input={0}".format(
        l_p_f(u_in.astype('float32')).shape))

    l_emb = lasagne.layers.ConcatLayer([l_emb, l_u_proc], axis=3)
    l_e_out = lasagne.layers.get_output(l_emb)
    l_o_f = theano.function([l_in.input_var, l_u_in.input_var], l_e_out)
    logger.info("output shape for emb+feature net={0}".format(
        l_o_f(x_in.astype('int32'), u_in.astype('float32')).shape))

    # -----------Replaced end ---------

    dropout_backward = lasagne.layers.DropoutLayer(l_emb, params['noise1'])
    dropout_forward = lasagne.layers.DropoutLayer(l_emb, params['noise1'])
    rnn_nonlinearity = lasagne.nonlinearities.tanh
    # rnn_nonlinearity = lasagne.nonlinearities.elu
    logger.info('Using RNN nonlinearity {0}'.format(rnn_nonlinearity))

    backward1 = lasagne.layers.LSTMLayer(dropout_backward, params['hidden1'], mask_input=l_mask, peepholes=False,
                                         forgetgate=lasagne.layers.Gate(b=lasagne.init.Constant(1.)),
                                         nonlinearity=rnn_nonlinearity, backwards=True, precompute_input=True,
                                         name='Backward-LSTM')
    forward1 = lasagne.layers.LSTMLayer(dropout_forward, params['hidden1'], mask_input=l_mask, peepholes=False,
                                        forgetgate=lasagne.layers.Gate(b=lasagne.init.Constant(1.)),
                                        nonlinearity=rnn_nonlinearity, precompute_input=True,
                                        name='Forward-LSTM')

    crf_layer = lasagne.layers.ConcatLayer([forward1, backward1], axis=2)
    dropout1 = lasagne.layers.DropoutLayer(crf_layer, p=params['noise1'])
    dimshuffle_layer = lasagne.layers.DimshuffleLayer(dropout1, (0, 2, 1))
    pairwise = lasagne.layers.Conv1DLayer(dimshuffle_layer, numTags * numTags, 2, name='LSTM convolution')
    pairwise = lasagne.layers.DimshuffleLayer(pairwise, (0, 2, 1))
    pairwise = lasagne.layers.batch_norm(pairwise, name='Convolution-normalization')

    reshape_for_dense = lasagne.layers.ReshapeLayer(
        dropout1, (-1, params['hidden1'] * 2))
    logger.info('Adding tanh non-linearity to accomodate CRF')
    output_nonlinearity = lasagne.nonlinearities.tanh
    dense1 = lasagne.layers.DenseLayer(reshape_for_dense, numTags, nonlinearity=output_nonlinearity,
                                       name='BI-LSTM')
    dense1 = lasagne.layers.batch_norm(dense1, name='BI-LSTM-Normalization')
    crf_layer = lasagne.layers.ReshapeLayer(dense1, (batch_size, params['maxlen'], numTags))

    return crf_layer, pairwise, l_in, l_mask, l_u_in