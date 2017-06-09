import matplotlib.pylab as plt
import bionlp.utils.monitoring as mo
import pickle
import os
import itertools
import numpy as np

SOURCE_DIR = "data/logs"


def load_monitoring(filename="monitor_2017-06-"):
    accuracy = {mo.TYPE_VALIDATION: [], mo.TYPE_TRAINING: []}
    loss = {mo.TYPE_VALIDATION: [], mo.TYPE_TRAINING: []}
    for file in os.listdir(SOURCE_DIR):
        if file.startswith(filename) and file.endswith('.pkl'):
            pkl = pickle.load(open(os.path.join(SOURCE_DIR, file), 'rb'))
            accuracy[mo.TYPE_VALIDATION].append(pkl[mo.TYPE_VALIDATION][mo.METRIC_ACC])
            accuracy[mo.TYPE_TRAINING].append(pkl[mo.TYPE_TRAINING][mo.METRIC_ACC])

            loss[mo.TYPE_VALIDATION].append(pkl[mo.TYPE_VALIDATION][mo.METRIC_LOSS_TOT])
            loss[mo.TYPE_TRAINING].append(pkl[mo.TYPE_TRAINING][mo.METRIC_LOSS_TOT])

    result = {mo.METRIC_ACC: accuracy, mo.METRIC_LOSS_TOT: loss}
    return result


def plot_monitoring_detail(monitoring_dict, metric):
    """ plots training and validation results, one graph for each run
    :param monitoring_dict:
    :param metric:
    :return:
    """
    assert metric in monitoring_dict, \
        "please use {0} of {1} as metric".format(mo.METRIC_ACC, load_monitoring(mo.METRIC_LOSS_TOT))
    data = monitoring_dict[metric]
    f = plt.figure()
    for idx, vals in enumerate(data[mo.TYPE_TRAINING]):
        ax = plt.subplot(plt.ceil(len(data['training'])/2), 2, idx + 1)
        ax.plot(vals, "-*", markersize=2)
        ax.plot(data[mo.TYPE_VALIDATION][idx], '-s', markersize=2)
        if idx + 1 > 8:
            ax.set_xlabel('Iteration number')
        if (idx + 1) % 2 == 1:
            ax.set_ylabel(metric.capitalize())

    return f


def plot_monitoring(monitoring_dict, metric, ax=None):
    """plots training or validation results, all runs in one graph
    :param monitoring_dict:
    :param metric:
    :param ax:
    :return:
    """
    assert metric in monitoring_dict, \
        "please use {0} of {1} as metric".format(mo.METRIC_ACC, load_monitoring(mo.METRIC_LOSS_TOT))

    training_data = monitoring_dict[metric][mo.TYPE_TRAINING]
    training_mean = mean_when_defined(training_data)

    validation_data = monitoring_dict[metric][mo.TYPE_VALIDATION]
    validation_mean = mean_when_defined(validation_data)

    if ax is None:
        ax = plt.subplot(1,1,1)

    for idx, training_values in enumerate(training_data):
        ax.plot(training_values, color="lightsteelblue")
        ax.plot(validation_data[idx], color="moccasin")
    training_handle, = ax.plot(training_mean, color="navy")
    validation_handle, = ax.plot(validation_mean, color="maroon")
    ax.legend([training_handle, validation_handle], ["Training", "Validation"])
    ax.set_xlabel("Iteration number")
    ax.set_ylabel(metric.capitalize())
    return ax


def parse_eval_line(line=""):
    fulltag, rest = line.split(' tag ')[1].split(' has ')
    tag = fulltag.replace("'", "")

    count, rest = rest.split(" elements ")
    recall, precision, f1 = rest.split("=")[1].split(",")

    return tag, {"count": int(count), "recall": float(recall), "precision": float(precision), "f1": float(f1)}


def load_eval(filename="eval_2017-06-"):
    result = {}
    for file in os.listdir(SOURCE_DIR):
        if file.startswith(filename) and file.endswith(".txt"):
            with open(os.path.join(SOURCE_DIR, file), "r") as fh:
                for line in fh:
                    if line.startswith("The tag"):
                        tag, line_result = parse_eval_line(line)

                        if tag not in result:
                            result[tag] = {"count": [], "recall": [], "precision": [], "f1": []}
                        [result[tag][key].append(value) for key, value in line_result.items()]
    return result


def plot_eval_detail(eval_dict):
    """ plots F1-scores vs number of training documents. One subplot for each tag
    :param eval_dict:
    :return:
    """
    f = plt.figure()
    idx = 1
    for tag, values in eval_dict.items():
        ax = plt.subplot(3, 3, idx)
        ax.plot(values['count'], values['f1'], '*b')
        # ax.plot(values['count'], values['recall'], 'og')
        # ax.plot(values['count'], values['precision'], '.r')
        ax.set_title(tag)
        if idx % 3 == 1:
            ax.set_ylabel('F1')
        if idx > 6:
            ax.set_xlabel('Count')
        idx += 1
    return f


def plot_eval_without_none(eval_dict, ax=None):
    markers = itertools.cycle(['s', '+', '.', 'o', '*', '^', '>', '<'])
    if ax is None:
        ax = plt.subplot(1,1,1)
    tags = []
    for tag, result in eval_dict.items():
        if tag != 'None':
            ax.plot(result['count'], result['f1'], next(markers))
            tags.append(tag)
    ax.legend(tags)
    ax.set_ylabel('F1-Score')


def plot_eval_none_only(eval_dict, ax=None):
    if ax is None:
        ax = plt.subplot(1,1,1)
    ax.plot(eval_dict['None']['count'], eval_dict['None']['f1'], 'v')
    ax.set_xlabel('# training documents')
    ax.set_ylabel('F1-Score')
    ax.legend(['None'])
    return ax


def plot_summary(monitorfile="eval_2017-06-", evalfile="eval_2017-06-"):
    monitor_dict = load_monitoring(monitorfile)
    eval_dict = load_eval(evalfile)

    f = plt.figure()
    eval_nonone_h = plt.subplot2grid((2, 2), (0, 0), colspan=2)
    eval_none_h = plt.subplot2grid((2, 2), (1, 0))
    monitor_h = plt.subplot2grid((2, 2), (1, 1))

    plot_eval_without_none(eval_dict, eval_nonone_h)
    plot_eval_none_only(eval_dict, eval_none_h)
    plot_monitoring(monitor_dict, mo.METRIC_ACC, monitor_h)

    eval_nonone_h.text(0.01, 0.9, 'A', transform=eval_nonone_h.transAxes, weight="bold", fontsize=16)
    eval_none_h.text(0.01, 0.9, 'B', transform=eval_none_h.transAxes, weight="bold", fontsize=16)
    monitor_h.text(0.01, 0.9, 'C', transform=monitor_h.transAxes, weight="bold", fontsize=16)
    return f


def mean_when_defined(data):
    """ compute the mean of the values in data, but include only those values that are defined
    The idea is to pad the missing values with nan and then use numpy.nanmean.
    Padding is done via the trick from https://stackoverflow.com/questions/32037893
    :param data:
    :return:
    """
    # Get lengths of each row of data
    lens = np.array([len(i) for i in data])

    # Mask of valid places in each row
    mask = np.arange(lens.max()) < lens[:,None]

    # Setup output array and put elements from data into masked positions
    padded_data = np.ones(mask.shape, dtype=float)*np.nan
    padded_data[mask] = np.concatenate(data)

    return np.nanmean(padded_data, axis=0)
