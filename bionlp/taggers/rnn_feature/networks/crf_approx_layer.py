from __future__ import print_function
import lasagne,sys
from bionlp.utils.utils import theano_logsumexp
import logging,theano
import theano.tensor as T

logging.basicConfig()
logger= logging.getLogger(__name__)


def constructApproximations(unary,pairwise,t_out,numTags,params,x_in,u_in,y_in,mask_in,l_in,l_mask,l_u_in,normalization=True):
    logger.info('Initializing Approximation layer with normalization as {0}'.format(normalization))
    #unary_potential =lasagne.layers.get_output(unary,deterministic=deterministic)
    #pairwise_potential =lasagne.layers.get_output(pairwise,deterministic=deterministic)
    forward_unary = lasagne.layers.SliceLayer(unary,indices=slice(0,params['maxlen']-1),axis=1)
    backward_unary = lasagne.layers.SliceLayer(unary,indices=slice(1,params['maxlen']),axis=1)
    forward_inputs = lasagne.layers.ConcatLayer([forward_unary,pairwise],axis=2)
    backward_inputs = lasagne.layers.ConcatLayer([backward_unary,pairwise],axis=2)
    crf_noise=0.0*params['noise1']/2.0
    logger.info('Noise used in approximate crf rnn layers is {0}'.format(crf_noise))
    backward_inputs=lasagne.layers.DropoutLayer(backward_inputs,crf_noise)
    forward_inputs=lasagne.layers.DropoutLayer(forward_inputs,crf_noise)

    forward1=lasagne.layers.RecurrentLayer(forward_inputs, numTags, W_in_to_hid=lasagne.init.GlorotUniform(), W_hid_to_hid=lasagne.init.GlorotUniform(), b=lasagne.init.Constant(0.), nonlinearity=lasagne.nonlinearities.rectify,hid_init=lasagne.init.Constant(0.))
    backward1=lasagne.layers.RecurrentLayer(backward_inputs, numTags, W_in_to_hid=lasagne.init.GlorotUniform(), W_hid_to_hid=lasagne.init.GlorotUniform(), b=lasagne.init.Constant(0.), nonlinearity=lasagne.nonlinearities.rectify,hid_init=lasagne.init.Constant(0.),backwards=True)

    '''
    forward1= lasagne.layers.GRULayer(forward_inputs,numTags,mask_input=l_mask,precompute_input=True)
    backward1= lasagne.layers.GRULayer(backward_inputs,numTags,mask_input=l_mask,backwards=True,precompute_input=True)
    '''

    forward_l=lasagne.regularization.regularize_layer_params(forward1,lasagne.regularization.l1)
    backward_l=lasagne.regularization.regularize_layer_params(backward1,lasagne.regularization.l1)

    def get_internal_results(deterministic):
        logger.info('Initializing Approx output  with normalization as {0} and deterministic as {1}'.format(normalization,deterministic))
        unary_potential =lasagne.layers.get_output(unary,deterministic=deterministic)
        forward_results=lasagne.layers.get_output(forward1,deterministic = deterministic)
        backward_results=lasagne.layers.get_output(backward1,deterministic = deterministic)
        backward_results=T.concatenate([backward_results,T.zeros_like(backward_results[:,:1,:])],axis=1)
        forward_results=T.concatenate([T.zeros_like(forward_results[:,:1,:]),forward_results],axis=1)


        unnormalized_prob = forward_results+unary_potential+backward_results
        marginal_results = theano_logsumexp(unnormalized_prob,axis=2)
        normalized_prob = unnormalized_prob - marginal_results.dimshuffle(0,1,'x')
        if normalization:
            return normalized_prob
        else:
            return unnormalized_prob

    training_p= get_internal_results(False)
    eval_p = get_internal_results(True)

    m_out = theano.function([l_in.input_var,l_u_in.input_var,l_mask.input_var],training_p)
    logger.debug('Log Prob size is {0}'.format(m_out(x_in.astype('int32'),u_in.astype('float32'),mask_in.astype('float32')).shape))
    approx_params=lasagne.layers.get_all_params([forward1,backward1],trainable=True)
    num_params = lasagne.layers.count_params([forward1,backward1],trainable=True)
    print('Number of parameters: {0}'.format(num_params))

    return training_p,eval_p,approx_params,forward_l+backward_l,[forward1,backward1]


def get_crf_training_loss(Inference_Layer,pairwise,t_out,numTags,params,x_in,y_in,mask_in,l_in,l_mask):

    pairwise_sequence=lasagne.layers.get_output(pairwise,deterministic = False)
    pairwise_sequence=pairwise_sequence.dimshuffle(1,0,2)
    pair_shape=pairwise_sequence.shape
    pairwise_potential=T.reshape(pairwise_sequence,(pair_shape[0],pair_shape[1],numTags,numTags))
    unary_potential = lasagne.layers.get_output(Inference_Layer,deterministic = False,unary=True)
    unnormalized_log = lasagne.layers.get_output(Inference_Layer,deterministic = False,unary=False)

    mask_out=lasagne.layers.get_output(l_mask)
    #Wp=Inference_Layer.get_CRF_params()
    logger.info('Initializing CRF structured loss')
    unary_score=T.sum(unary_potential*t_out*mask_out.dimshuffle(0,1,'x'),axis=[1,2])
    unary_sequence = t_out.dimshuffle(1,0,2)    #Reshuffling the batched unary potential shape so that it can be used for word level iterations in theano.scan
    mask_out=mask_out.dimshuffle(1,0)
    m_out_f=theano.function([l_mask.input_var],mask_out)
    print("rearranged output shape for mask",m_out_f(mask_in.astype('float32')).shape)

    def pairwise_collector(yprev,ycurrent,Wp):
        yip = yprev.dimshuffle(0,1,'x')*Wp
        yip = yip*ycurrent.dimshuffle(0,'x',1)
        yip = T.sum(yip,axis=[1,2])
        return yip


    pairwise_score,_=theano.scan(fn=pairwise_collector,sequences=[dict(input=unary_sequence,taps=[-1,0]),dict(input=pairwise_potential,taps=[0])],non_sequences=None)
    pairwise_score_result = theano.function([t_out,l_in.input_var,l_mask.input_var],pairwise_score)
    logger.debug('Scan pairwise score is calculated.{0}'.format(pairwise_score_result(y_in.astype('float32'),x_in.astype('int32'),mask_in.astype('float32')).shape))
    pairwise_score=T.sum(pairwise_score*mask_out[1:,:],axis=[0])
    total_score=theano.function([t_out,l_in.input_var,l_mask.input_var],pairwise_score+unary_score)
    zee_ = theano.function([l_in.input_var,l_mask.input_var],theano_logsumexp(unnormalized_log,axis=2)[:,0])
    cost_ = theano.function([t_out,l_in.input_var,l_mask.input_var],theano_logsumexp(unnormalized_log,axis=2)[:,0]-(pairwise_score+unary_score))

    logger.debug('Sentence score is calculated.The vector size {0} should be the same as the batch size.'.format(total_score(y_in.astype('float32'),x_in.astype('int32'),mask_in.astype('float32')).shape))
    logger.debug('Zee score is calculated.The vector size {0} should be the same as the batch size.'.format(zee_(x_in.astype('int32'),mask_in.astype('float32'))))
    logger.debug('Cost score is calculated.The vector size {0} should be the same as the batch size.'.format(cost_(y_in.astype('float32'),x_in.astype('int32'),mask_in.astype('float32')).shape))

    #sys.exit(0)
    return theano_logsumexp(unnormalized_log,axis=2)[:,0]-(pairwise_score+unary_score)


