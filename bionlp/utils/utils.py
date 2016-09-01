import logging,lasagne
import numpy as np
import theano.tensor as T
logging.basicConfig()
logger= logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)

# adapted from https://github.com/nouiz/lisa_emotiw/blob/master/emotiw/wardefar/crf_theano.py
def theano_logsumexp(x, axis=None):
    """
    Compute log(sum(exp(x), axis=axis) in a numerically stable
    fashion.
    Parameters
    ----------
    x : tensor_like
        A Theano tensor (any dimension will do).
    axis : int or symbolic integer scalar, or None
        Axis over which to perform the summation. `None`, the
        default, performs over all axes.
    Returns
    -------
    result : ndarray or scalar
        The result of the log(sum(exp(...))) operation.
    """
    xmax = T.max(x,axis=axis, keepdims=True)
    xmax_ = T.max(x,axis=axis)
    return xmax_ + T.log(T.exp(x - xmax).sum(axis=axis))

