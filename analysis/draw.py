import matplotlib.pylab as plt
import itertools
import numpy as np
import analysis.io as io
import bionlp.utils.monitoring as mo
import analysis.utils as utils


def plot_monitoring_detail(monitoring_dict, metric):
    """ plots training and validation results, one graph for each run
    :param monitoring_dict:
    :param metric:
    :return:
    """
    assert metric in monitoring_dict, \
        "please use {0} of {1} as metric".format(mo.METRIC_ACC, io.load_monitoring(mo.METRIC_LOSS_TOT))
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
    :param metric: 'accuracy' or 'loss'
    :param ax: handle to the axes to plot into. If None, a new figure will be created
    :return:
    """
    assert metric in monitoring_dict, \
        "please use {0} of {1} as metric".format(mo.METRIC_ACC, io.load_monitoring(mo.METRIC_LOSS_TOT))

    training_data = monitoring_dict[metric][mo.TYPE_TRAINING]
    training_mean = utils.mean_when_defined(training_data)

    validation_data = monitoring_dict[metric][mo.TYPE_VALIDATION]
    validation_mean = utils.mean_when_defined(validation_data)

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


def plot_eval_detail(eval_dict):
    """ plots F1-scores vs number of training documents for several runs. One subplot for each tag
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
    """ plots F1-scores vs number of training documents for several runs, omit the None tag
    """
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
    """ plots F1-scores vs number of training documents for several runs, only the None tag
        """
    if ax is None:
        ax = plt.subplot(1,1,1)
    ax.plot(eval_dict['None']['count'], eval_dict['None']['f1'], 'v')
    ax.set_xlabel('# training documents')
    ax.set_ylabel('F1-Score')
    ax.legend(['None'])
    return ax


def plot_metrics(confusion_metrics, ax=None):
    if ax is None:
        ax = plt.subplot(1,1,1)
    x_ticks = np.arange(len(confusion_metrics['f_score']))
    ax.barh(x_ticks, confusion_metrics['f_score'])
    ax.set_yticks(x_ticks)
    ax.set_yticklabels(confusion_metrics['tags'])


def plot_multirun_summary(monitorfile="eval_2017-06-", evalfile="eval_2017-06-"):
    monitor_dict = io.load_monitoring(monitorfile)
    eval_dict = io.load_eval_txt_file(evalfile)

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


def plot_cv_summary(monitorfile, evalfile):
    monitor_dict = io.load_monitoring(monitorfile)
    cm = io.load_confusion_matrix(evalfile)
    confusion_metrics = utils.cm_to_metrics(cm)

    f = plt.figure()
    eval_h = plt.subplot(2, 1, 1)
    monitor_h = plt.subplot(2, 1, 2)

    plot_metrics(confusion_metrics, eval_h)
    plot_monitoring(monitor_dict, mo.METRIC_ACC, monitor_h)

