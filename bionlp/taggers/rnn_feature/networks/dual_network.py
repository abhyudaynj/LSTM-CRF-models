import numpy as np

from crf_dual_layer import DualCRFLayer,get_crf_training_loss
import lasagne,time,theano,logging,sys
import theano.tensor as T
from bionlp.utils.utils import theano_logsumexp
logging.basicConfig()
logger= logging.getLogger(__name__)


def setup_NN(worker,x_in,u_in,mask_in,y_in,params,numTags,emb_w):
    logger.info('Loading CRF-RNN network format with pairwise modeling. Forward Backward will be used for calculating the marginals while training. The output sequence will be calculated using posterior decoding.')
    batch_size = x_in.shape[0]
    print('X train batch size {0}'.format(x_in.shape))
    print('U train batch size {0}'.format(u_in.shape))
    print('Y train batch size {0}'.format(y_in.shape))
    print('mask train batch size {0}'.format(mask_in.shape))
    xt=theano.shared(x_in)
    mt=theano.shared(mask_in)
    ut=theano.shared(u_in)
    yt=theano.shared(y_in)
    premodel=time.time()

    l_in=lasagne.layers.InputLayer(shape=(None,params['maxlen'],1),input_var=xt.astype('int32'))
    l_u_in=lasagne.layers.InputLayer(shape=(None,params['maxlen'],u_in.shape[2]))
    l_mask=lasagne.layers.InputLayer(shape=(None,params['maxlen']))
    #---------- Replaced -------------

    l_u_reshape=lasagne.layers.ReshapeLayer(l_u_in,(-1,u_in.shape[2]))
    l_e_out=lasagne.layers.get_output(l_u_reshape)
    l_o_f = theano.function([l_u_in.input_var],l_e_out)
    logger.info("output shape for reshaped feature net = {0}".format(l_o_f(u_in.astype('float32')).shape))
    logger.info('Using {0} dimensional layer for extra features'.format(params['feature1']))
    l_u_dense=lasagne.layers.DenseLayer(l_u_reshape,params['feature1'])
    l_u_proc=lasagne.layers.ReshapeLayer(l_u_dense,(batch_size,params['maxlen'],1,params['feature1']))
    if params['word2vec']==1:
        l_emb=lasagne.layers.EmbeddingLayer(l_in,emb_w.shape[0],emb_w.shape[1],W=emb_w.astype('float32'))
        l_emb.add_param(l_emb.W, l_emb.W.get_value().shape, trainable=params['emb1'])
    else:
        l_emb=lasagne.layers.EmbeddingLayer(l_in,emb_w.shape[0],emb_w.shape[1])


    if params['emb2']>0:
        l_emb1=lasagne.layers.EmbeddingLayer(l_in,emb_w.shape[0],params['emb2'])
        l_emb=lasagne.layers.ConcatLayer([l_emb,l_emb1],axis=3)

    l_e_out=lasagne.layers.get_output(l_emb)
    l_p_out=lasagne.layers.get_output(l_u_proc)
    l_o_f = theano.function([l_in.input_var],l_e_out)
    l_p_f = theano.function([l_u_in.input_var],l_p_out)
    logger.info("output shape for emb net ={0}".format(l_o_f(x_in.astype('int32')).shape))
    logger.info("output shape for feature input={0}".format(l_p_f(u_in.astype('float32')).shape))


    l_emb=lasagne.layers.ConcatLayer([l_emb,l_u_proc],axis=3)
    l_e_out=lasagne.layers.get_output(l_emb)
    l_o_f = theano.function([l_in.input_var,l_u_in.input_var],l_e_out)
    logger.info("output shape for emb+feature net={0}".format(l_o_f(x_in.astype('int32'),u_in.astype('float32')).shape))

    #-----------Replaced end ---------


    dropout_backward=lasagne.layers.DropoutLayer(l_emb,params['noise1'])
    dropout_forward=lasagne.layers.DropoutLayer(l_emb,params['noise1'])

    rnn_nonlinearity = lasagne.nonlinearities.tanh
    #rnn_nonlinearity = lasagne.nonlinearities.elu
    logger.info('Using RNN nonlinearity {0}'.format(rnn_nonlinearity))

    backward1= lasagne.layers.LSTMLayer(dropout_backward,params['hidden1'],mask_input=l_mask,peepholes=False,forgetgate=lasagne.layers.Gate(b=lasagne.init.Constant(1.)),nonlinearity=rnn_nonlinearity,backwards=True,precompute_input=True)
    forward1= lasagne.layers.LSTMLayer(dropout_forward,params['hidden1'],mask_input=l_mask,peepholes=False,forgetgate=lasagne.layers.Gate(b=lasagne.init.Constant(1.)),nonlinearity=rnn_nonlinearity,precompute_input=True)

    crf_layer=lasagne.layers.ConcatLayer([forward1,backward1],axis=2)


    mid_out=lasagne.layers.get_output(crf_layer,deterministic=True)
    mid_output = theano.function([l_in.input_var,l_u_in.input_var,l_mask.input_var],mid_out)
    logger.info("output shape for for unary mid layer {0}".format(mid_output(x_in.astype('int32'),u_in.astype('float32'),mask_in.astype('float32')).shape))
    logger.info("output sum for for unary mid layer {0}".format(np.sum(mid_output(x_in.astype('int32'),u_in.astype('float32'),mask_in.astype('float32'))[0,:,:],axis=1)))


    dropout1=lasagne.layers.DropoutLayer(crf_layer,p=params['noise1'])
    dimshuffle_layer=lasagne.layers.DimshuffleLayer(dropout1,(0,2,1))
    pairwise = lasagne.layers.Conv1DLayer(dimshuffle_layer,numTags*numTags,2)
    pairwise = lasagne.layers.DimshuffleLayer(pairwise,(0,2,1))
    pairwise = lasagne.layers.batch_norm(pairwise)

    reshape_for_dense=lasagne.layers.ReshapeLayer(dropout1,(-1,params['hidden1']*2))
    output_nonlinearity=lasagne.nonlinearities.tanh
    dense1=lasagne.layers.DenseLayer(reshape_for_dense,numTags,nonlinearity=output_nonlinearity)
    dense1 = lasagne.layers.batch_norm(dense1)
    crf_layer=lasagne.layers.ReshapeLayer(dense1,(batch_size,params['maxlen'],numTags))

    mid_out=lasagne.layers.get_output(crf_layer,deterministic=True)
    mid_output = theano.function([l_in.input_var,l_u_in.input_var,l_mask.input_var],mid_out)
    logger.info("output shape for for unary mid layer {0}".format(mid_output(x_in.astype('int32'),u_in.astype('float32'),mask_in.astype('float32')).shape))
    logger.info("output sum for for unary mid layer {0}".format(np.sum(mid_output(x_in.astype('int32'),u_in.astype('float32'),mask_in.astype('float32'))[0,:,:],axis=1)))

    sum_layer=DualCRFLayer([crf_layer,pairwise,l_mask],mask_input=True)

    unaryPotential = lasagne.layers.get_output(crf_layer,deterministic = False)
    outp=lasagne.layers.get_output(sum_layer,deterministic = False)
    eval_out=lasagne.layers.get_output(sum_layer,deterministic=True)
    crf_output = theano.function([l_in.input_var,l_u_in.input_var,l_mask.input_var],eval_out)

    lstm_output = crf_output # Included for future functionality
    print("output shape for theano net",lstm_output(x_in.astype('int32'),u_in.astype('float32'),mask_in.astype('float32')).shape)
    t_out=T.tensor3()
    eval_cost=T.mean((eval_out-t_out)**2)

    all_params = lasagne.layers.get_all_params(sum_layer,trainable=True)
    print('Params :{0}'.format(all_params))
    logger.info('\'l1\' and \'l2\' Regularization is applied to all trainable parameters in the network')
    l2_cost= params['l2'] *lasagne.regularization.apply_penalty(all_params,lasagne.regularization.l2)
    l1_cost= params['l1'] *lasagne.regularization.apply_penalty(all_params,lasagne.regularization.l1)
    regularization_losses=l2_cost+l1_cost


    num_params = lasagne.layers.count_params(sum_layer,trainable = True)
    print('Number of parameters: {0}'.format(num_params))

    crf_train_loss=get_crf_training_loss(sum_layer,pairwise,t_out,numTags,params,x_in,u_in,y_in,mask_in,l_in,l_u_in,l_mask)
    cost=T.mean(crf_train_loss)+ regularization_losses
    cost_loss=T.mean(crf_train_loss)
    cost_regularization= regularization_losses


    updates = lasagne.updates.adagrad(cost, all_params,learning_rate=params['learning-rate'])
    #updates = lasagne.updates.adam(cost, all_params,learning_rate=params['learning-rate'])
    if params['momentum']==2:
        updates = lasagne.updates.apply_nesterov_momentum(updates,all_params,momentum=0.9)
    if params['momentum']==1:
        updates = lasagne.updates.apply_momentum(updates,all_params,momentum=0.9)

    train = theano.function([l_in.input_var,l_u_in.input_var,t_out,l_mask.input_var],cost,updates=updates)

    compute_cost = theano.function([l_in.input_var,l_u_in.input_var,t_out,l_mask.input_var],cost)
    compute_cost_loss = theano.function([l_in.input_var,l_u_in.input_var,t_out,l_mask.input_var],cost_loss)
    compute_cost_regularization = theano.function([],cost_regularization)
    acc_=T.sum(T.eq(T.argmax(eval_out,axis=2),T.argmax(t_out,axis=2))*l_mask.input_var)/T.sum(l_mask.input_var)
    compute_acc = theano.function([l_in.input_var,l_u_in.input_var,t_out,l_mask.input_var],acc_)
    print('Time to build and compile model {0}'.format(time.time()-premodel))


    return {'crf_output':crf_output,'lstm_output':lstm_output,'train':train,'compute_cost':compute_cost,'compute_acc':compute_acc,'compute_cost_loss':compute_cost_loss,'compute_cost_regularization':compute_cost_regularization,'final_layers':sum_layer}

