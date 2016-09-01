from __future__ import print_function
from bionlp.utils.utils import theano_logsumexp
import lasagne,sys
import logging,theano
import theano.tensor as T

logging.basicConfig()
logger= logging.getLogger(__name__)

class DualCRFLayer(lasagne.layers.MergeLayer):
    def __init__(self,incomings,mask_input=None,**kwargs):
        super(DualCRFLayer,self).__init__(incomings, **kwargs)
        logger.info('Initializing DualCRF inference layer')
        self.num_labels=incomings[0].output_shape[-1] #getting the last dimension which is the number of labels.
        if mask_input==True:
            self.mask_input = True
        else:
            assert False,'Output without Mask is not yet Enabled'
        assert incomings.__len__()==3,'the CRF layer should have 3 inputs: unary_potential,pairwise_potential and mask'

    def get_output_shape_for(self, input_shapes):
        return input_shapes[0]

    def get_output_for(self,inputs,normalization=False,**kwargs):
        net_input = inputs[0]
        pairwise_sequence=inputs[1].dimshuffle(1,0,2)
        assert inputs.__len__()==3,'the CRF layer should have 3 inputs: unary_potential,pairwise_potential and mask'
        mask=inputs[2]
        pair_shape=pairwise_sequence.shape
        pairwise_sequence=T.reshape(pairwise_sequence,(pair_shape[0],pair_shape[1],self.num_labels,self.num_labels))
        if 'unary' in kwargs and kwargs['unary']==True:
            return net_input[0]


        unary_sequence = net_input.dimshuffle(1,0,2)    #Reshuffling the batched unary potential shape so that it can be used for word level iterations in theano.scan

        mask = mask.dimshuffle(1,0)
        def forward_scan1(unary_sequence,Wp,mask,forward_sm):
            forward_sm=forward_sm+unary_sequence*mask.dimshuffle(0,'x')
            forward_sm=theano_logsumexp(forward_sm.dimshuffle(0,1,'x')+mask.dimshuffle(0,'x','x')*Wp,1)
            return forward_sm

        def backward_scan1(unary_sequence,Wp,mask,forward_sm):
            forward_sm=forward_sm+unary_sequence*mask.dimshuffle(0,'x')
            forward_sm=theano_logsumexp(forward_sm.dimshuffle(0,1,'x')+mask.dimshuffle(0,'x','x')*Wp.dimshuffle(0,2,1),1)
            return forward_sm

        forward_results,_=theano.scan(fn=forward_scan1,sequences=[unary_sequence,pairwise_sequence,mask],outputs_info=T.zeros_like(unary_sequence[0]),non_sequences=None,n_steps=unary_sequence.shape[0]-1)
        backward_results,_=theano.scan(fn=backward_scan1,sequences=[unary_sequence[::-1],pairwise_sequence[::-1],mask[::-1]],outputs_info=T.zeros_like(unary_sequence[0]),non_sequences=None,n_steps=unary_sequence.shape[0]-1)

        backward_results=T.concatenate([backward_results[::-1],T.zeros_like(backward_results[:1])],axis=0)
        forward_results=T.concatenate([T.zeros_like(forward_results[:1]),forward_results],axis=0)

        unnormalized_prob = forward_results+unary_sequence+backward_results
        marginal_results = theano_logsumexp(unnormalized_prob,axis=2)
        normalized_prob = unnormalized_prob - marginal_results.dimshuffle(0,1,'x')
        # provided for debugging purposes.
        #marginal_all = theano.function([l_in.input_var,l_mask.input_var],marginal_results)
        #probs=theano.function([l_in.input_var,l_mask.input_var],normalized_prob.dimshuffle(1,0,2))
        if normalization:
            return normalized_prob.dimshuffle(1,0,2)
        else:
            return unnormalized_prob.dimshuffle(1,0,2)


def get_crf_training_loss(Inference_Layer,pairwise,t_out,numTags,params,x_in,u_in,y_in,mask_in,l_in,l_u_in,l_mask):

    pairwise_sequence=lasagne.layers.get_output(pairwise,deterministic = False)
    pairwise_sequence=pairwise_sequence.dimshuffle(1,0,2)
    pair_shape=pairwise_sequence.shape
    pairwise_potential=T.reshape(pairwise_sequence,(pair_shape[0],pair_shape[1],numTags,numTags))
    unary_potential = lasagne.layers.get_output(Inference_Layer,deterministic = False,unary=True)
    unnormalized_log = lasagne.layers.get_output(Inference_Layer,deterministic = False,unary=False)

    mask_out=lasagne.layers.get_output(l_mask)
    logger.info('Initializing CRF structured loss')
    unary_score=T.sum(unary_potential*t_out*mask_out.dimshuffle(0,1,'x'),axis=[1,2])
    unary_sequence = t_out.dimshuffle(1,0,2)    #Reshuffling the batched unary potential shape so that it can be used for word level iterations in theano.scan
    mask_out=mask_out.dimshuffle(1,0)
    m_out_f=theano.function([l_mask.input_var],mask_out)
    logger.info("rearranged output shape for mask {0}".format(m_out_f(mask_in.astype('float32')).shape))

    def pairwise_collector(yprev,ycurrent,Wp):
        yip = yprev.dimshuffle(0,1,'x')*Wp
        yip = yip*ycurrent.dimshuffle(0,'x',1)
        yip = T.sum(yip,axis=[1,2])
        return yip


    pairwise_score,_=theano.scan(fn=pairwise_collector,sequences=[dict(input=unary_sequence,taps=[-1,0]),dict(input=pairwise_potential,taps=[0])],non_sequences=None)
    pairwise_score_result = theano.function([t_out,l_in.input_var,l_u_in.input_var,l_mask.input_var],pairwise_score)
    logger.debug('Scan pairwise score is calculated.{0}'.format(pairwise_score_result(y_in.astype('float32'),x_in.astype('int32'),u_in.astype('float32'),mask_in.astype('float32')).shape))
    pairwise_score=T.sum(pairwise_score*mask_out[1:,:],axis=[0])
    total_score=theano.function([t_out,l_in.input_var,l_u_in.input_var,l_mask.input_var],pairwise_score+unary_score)
    zee_ = theano.function([l_in.input_var,l_u_in.input_var,l_mask.input_var],theano_logsumexp(unnormalized_log,axis=2))
    up_ = theano.function([l_in.input_var,l_u_in.input_var,l_mask.input_var],unnormalized_log)
    cost_ = theano.function([t_out,l_in.input_var,l_u_in.input_var,l_mask.input_var],theano_logsumexp(unnormalized_log,axis=2)[:,0]-(pairwise_score+unary_score))

    logger.debug('Unary potential score is calculated.The vector size {0} should be the same as the batch size.'.format(up_(x_in.astype('int32'),u_in.astype('float32'),mask_in.astype('float32')).shape))
    logger.debug('Sentence score is calculated.The vector size {0} should be the same as the batch size.'.format(total_score(y_in.astype('float32'),x_in.astype('int32'),u_in.astype('float32'),mask_in.astype('float32')).shape))
    visible_zee=zee_(x_in.astype('int32'),u_in.astype('float32'),mask_in.astype('float32'))
    logger.debug('Zee score shape is calculated.The vector size {0} should be the same as the batch size.'.format(visible_zee.shape))
    logger.debug('Zee score is calculated for sample 0. Zee should be the same for all label instances. \nZee :{0} \nMask: {1}'.format(visible_zee[0,:][mask_in[0]!=0],mask_in[0]))
    #logger.debug('Zee score shape for sample 0 {0}'.format(visible_zee[0,:].shape))
    logger.debug('Zee score is calculated for sample 10. Zee should be the same for all label instances. \nZee :{0} \nMask: {1}'.format(visible_zee[10,:][mask_in[10]!=0],mask_in[10]))
    #logger.debug('Zee score shape for sample 11 {0}'.format(visible_zee[11,:].shape))
    logger.debug('Zee score across the sample batch.This should not be same for all batch instances unless there is no embedding initialization.{0}'.format(zee_(x_in.astype('int32'),u_in.astype('float32'),mask_in.astype('float32'))[:,0]))
    logger.debug('Cost score is calculated.The vector size {0} should be the same as the batch size.'.format(cost_(y_in.astype('float32'),x_in.astype('int32'),u_in.astype('float32'),mask_in.astype('float32')).shape))

    return theano_logsumexp(unnormalized_log,axis=2)[:,0]-(pairwise_score+unary_score)


