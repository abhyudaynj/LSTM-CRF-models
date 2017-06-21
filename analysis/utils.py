import numpy as np


def mean_when_defined(data):
    """ compute the mean of the values in data, where data is a list of lists,
    but include only those values that are defined.
    The idea is to pad the missing values with nan and then use numpy.nanmean.
    Padding is done via the trick from https://stackoverflow.com/questions/32037893
    :param data:
    :return:
    """
    # Get lengths of each row of data
    lens = np.array([len(i) for i in data])

    # Mask of valid places in each row
    mask = np.arange(lens.max()) < lens[:, None]

    # Setup output array and put elements from data into masked positions
    padded_data = np.ones(mask.shape, dtype=float)*np.nan
    padded_data[mask] = np.concatenate(data)

    return np.nanmean(padded_data, axis=0)


def cm_to_metrics(cm):
    """

    :param cm: nltk.metric.confusionmatrix instance
    :return: actual_counts, precision, recall, f_score
    """
    tags = np.array(cm._values)
    matrix = np.array(cm._confusion)
    actual_counts = np.sum(matrix, axis=1)
    predicted_counts = np.sum(matrix, axis=0)
    true_positives = np.diagonal(matrix)

    precision = true_positives / predicted_counts
    recall = true_positives / actual_counts
    f_score = 2 * precision * recall / (precision + recall)

    p = np.argsort(actual_counts)
    print(p)

    return {'tags': tags[p], 'counts': actual_counts[p], 'precision': precision[p], 'recall': recall[p], 'f_score': f_score[p]}
