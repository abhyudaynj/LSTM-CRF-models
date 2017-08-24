from bionlp.taggers.rnn_feature.networks.approx_network import setup_NN as approx_NN
from bionlp.taggers.rnn_feature.networks.network import setup_NN as unary_NN
from bionlp.taggers.rnn_feature.networks.dual_network import setup_NN as dual_NN
import logging
import json


logger = logging.getLogger(__name__)


def get_crf_model(params):
    if 'CRF_MODEL_ON' in params and params['CRF_MODEL_ON']:
        logger.info('CRF IS ON. CRF_MODELS WILL BE USED')
        if params['mode'] == 1:
            setup_nn = approx_NN
            logger.info('MODE :Using the Approximate Message Passing framework')
        elif params['mode'] == -1:
            setup_nn = unary_NN
            logger.info('MODE : Modeling only the unary potentials')
        else:
            setup_nn = dual_NN
            logger.info('MODE : Modeling both unary and binary potentials')
    else:
        logger.info(
            'CRF IS NOT ON. This tagger only supports CRF models. A default CRF_MODEL will be used.')
        params['mode'] = 1
        params['CRF_MODEL_ON'] = True
        setup_nn = approx_NN
        logger.info('MODE :Using the Approximate Message Passing framework')

    logger.info('Using the parameters:\n {0}'.format(json.dumps(params, indent=2)))
    return setup_nn
