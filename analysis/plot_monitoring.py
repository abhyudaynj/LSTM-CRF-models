import matplotlib.pylab as plt
import bionlp.utils.monitoring as mo
import pickle
import os

SOURCE_DIR = "data/logs"

# f = plt.figure()
# ah = f.get_axes()


def load_monitoring(filename="monitor_2017-06-"):
    accuracy = {mo.TYPE_VALIDATION: [], mo.TYPE_TRAINING: []}
    loss = {mo.TYPE_VALIDATION: [], mo.TYPE_TRAINING: []}
    for file in os.listdir(SOURCE_DIR):
        if file.startswith(filename) and file.endswith('.pkl'):
            pkl = pickle.load(open(os.path.join(SOURCE_DIR, file), 'rb'))
            accuracy[mo.TYPE_VALIDATION].append(pkl[mo.TYPE_VALIDATION][mo.KEY_ACC])
            accuracy[mo.TYPE_TRAINING].append(pkl[mo.TYPE_TRAINING][mo.KEY_ACC])

            loss[mo.TYPE_VALIDATION].append(pkl[mo.TYPE_VALIDATION][mo.KEY_LOSS_TOT])
            loss[mo.TYPE_TRAINING].append(pkl[mo.TYPE_TRAINING][mo.KEY_LOSS_TOT])

    return accuracy, loss


def plot_monitoring(data):
    f = plt.figure()
    for iter, vals in enumerate(data['training']):
        ax = plt.subplot(plt.ceil(len(data['training'])/2), 2, iter + 1)
        ax.plot(vals,"-*", markersize=2)
        ax.plot(data['validation'][iter], '-s', markersize=2)
        if iter + 1 > 8:
            ax.set_xlabel('Iteration number')
        if (iter + 1) % 2 == 1:
            ax.set_ylabel('Accuracy')

    return f


def parse_line(line=""):
    fulltag, rest = line.split(' tag ')[1].split(' has ')
    tag = fulltag.replace("'","")

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
                        tag, line_result = parse_line(line)

                        if tag not in result:
                            result[tag] = {"count": [], "recall": [], "precision": [], "f1": []}
                        [result[tag][key].append(value) for key, value in line_result.items()]
    return result


def plot_eval(filename="eval_2017-06-"):
    result = load_eval(filename)

    f = plt.figure()
    idx = 1
    for tag, values in result.items():
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



