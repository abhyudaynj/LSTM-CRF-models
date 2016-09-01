from __future__ import print_function
from bionlp.utils.utils import theano_logsumexp
import lasagne,sys
import logging,theano
import theano.tensor as T

logging.basicConfig()
logger= logging.getLogger(__name__)

class CRFLayer(lasagne.layers.Layer):
    def __init__(self,incoming,Wp=lasagne.init.GlorotNormal(),**kwargs):
        super(CRFLayer,self).__init__(incoming, **kwargs)
        logger.info('Initializing CRF inference layer')
        self.num_labels=self.input_shape[-1] #getting the last dimension which is the number of labels.
        self.W= self.add_param(Wp, (self.num_labels,self.num_labels),name ='Wp')

    def get_CRF_params(self):
        return self.W

    def get_output_for(self,net_input,**kwargs):
        if 'unary' in kwargs and kwargs['unary']==True:
            return net_input

        logger.info('Initializing the messages')
        Wp=self.W
        unary_sequence = net_input.dimshuffle(1,0,2)    #Reshuffling the batched unary potential shape so that it can be used for word level iterations in theano.scan

        def forward_scan1(unary_sequence,forward_sm,Wp):
            forward_sm=forward_sm+unary_sequence
            forward_sm=theano_logsumexp(forward_sm.dimshuffle(0,1,'x')+Wp,1)
            return forward_sm

        def backward_scan1(unary_sequence,forward_sm,Wp):
            forward_sm=forward_sm+unary_sequence
            forward_sm=theano_logsumexp(forward_sm.dimshuffle(0,1,'x')+Wp.T,1)
            return forward_sm


        forward_results,_=theano.scan(fn=forward_scan1,sequences=[unary_sequence],outputs_info=T.zeros_like(unary_sequence[0]),non_sequences=[Wp],n_steps=unary_sequence.shape[0]-1)
        backward_results,_=theano.scan(fn=backward_scan1,sequences=[unary_sequence[::-1]],outputs_info=T.zeros_like(unary_sequence[0]),non_sequences=[Wp],n_steps=unary_sequence.shape[0]-1)

        backward_results=T.concatenate([backward_results[::-1],T.zeros_like(backward_results[:1])],axis=0)
        forward_results=T.concatenate([T.zeros_like(forward_results[:1]),forward_results],axis=0)

        unnormalized_prob = forward_results+unary_sequence+backward_results
        marginal_results = theano_logsumexp(unnormalized_prob,axis=2)
        normalized_prob = unnormalized_prob - marginal_results.dimshuffle(0,1,'x')
        # provided for debugging purposes.
        #marginal_all = theano.function([l_in.input_var,l_mask.input_var],marginal_results)
        #probs=theano.function([l_in.input_var,l_mask.input_var],normalized_prob.dimshuffle(1,0,2))
        if 'normalized' in kwargs and kwargs['normalized']==True:
            return normalized_prob.dimshuffle(1,0,2)
        else:
            return unnormalized_prob.dimshuffle(1,0,2)


def get_crf_training_loss(Inference_Layer,t_out,numTags,params,x_in,u_in,y_in,mask_in,l_in,l_u_in,l_mask):

    unary_potential = lasagne.layers.get_output(Inference_Layer,deterministic = False,unary=True)
    unnormalized_log = lasagne.layers.get_output(Inference_Layer,deterministic = False,unary=False)
    Wp=Inference_Layer.get_CRF_params()
    logger.info('Initializing CRF structured loss')
    unary_score=T.sum(unary_potential*t_out,axis=[1,2])
    unary_sequence = t_out.dimshuffle(1,0,2)    #Reshuffling the batched unary potential shape so that it can be used for word level iterations in theano.scan

    def pairwise_collector(yprev,ycurrent,Wp):
        yip = yprev.dimshuffle(0,1,'x')*Wp.dimshuffle('x',0,1)
        yip = yip*ycurrent.dimshuffle(0,'x',1)
        yip = T.sum(yip,axis=[1,2])
        return yip


    pairwise_score,_=theano.scan(fn=pairwise_collector,sequences=[dict(input=unary_sequence,taps=[-1,0])],non_sequences=[Wp])
    pairwise_score_result = theano.function([t_out],pairwise_score)
    logger.debug('Scan pairwise score is calculated.{0}'.format(pairwise_score_result(y_in.astype('float32')).shape))
    pairwise_score=T.sum(pairwise_score,axis=[0])
    total_score=theano.function([t_out,l_in.input_var,l_u_in.input_var,l_mask.input_var],pairwise_score+unary_score)
    zee_ = theano.function([l_in.input_var,l_u_in.input_var,l_mask.input_var],theano_logsumexp(unnormalized_log,axis=2))
    cost_ = theano.function([t_out,l_in.input_var,l_u_in.input_var,l_mask.input_var],theano_logsumexp(unnormalized_log,axis=2)[:,0]-(pairwise_score+unary_score))

    logger.debug('Sentence score is calculated.The vector size {0} should be the same as the batch size.'.format(total_score(y_in.astype('float32'),x_in.astype('int32'),u_in.astype('float32'),mask_in.astype('float32')).shape))
    logger.debug('Zee score is calculated.The vector size {0} should be the same as the batch size.'.format(zee_(x_in.astype('int32'),u_in.astype('float32'),mask_in.astype('float32')).shape))
    logger.debug('Zee is calculated for sample 0. This should be same for all label instances.{0}'.format(zee_(x_in.astype('int32'),u_in.astype('float32'),mask_in.astype('float32'))[0,:][mask_in[0]!=0]))
    logger.debug('Zee is calculated for sample 10. This should be same for all label instances.{0}'.format(zee_(x_in.astype('int32'),u_in.astype('float32'),mask_in.astype('float32'))[10,:][mask_in[10]!=0]))
    logger.debug('Zee score across the sample batch. This should not be same for all batch instances unless there is no embedding initialization.{0}'.format(zee_(x_in.astype('int32'),u_in.astype('float32'),mask_in.astype('float32'))[:,0]))
    logger.debug('Cost score is calculated.The vector size {0} should be the same as the batch size.'.format(cost_(y_in.astype('float32'),x_in.astype('int32'),u_in.astype('float32'),mask_in.astype('float32')).shape))

    return theano_logsumexp(unnormalized_log,axis=2)[:,0]-(pairwise_score+unary_score)


