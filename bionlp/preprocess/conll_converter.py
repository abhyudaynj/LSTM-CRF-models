import os
import json
import argparse

SOURCE_DIR = "../../data/sources/CoNLL_2000"
TARGET_DIR = "../../data/converted/CoNLL_2000/datasets"
SOURCE_FILE = "train.txt"
BATCH_SIZE = 1000


def write_files(sentences, annotations, filename):
    with open(os.path.join(TARGET_DIR, filename), 'w') as txt_file:
        txt_file.write("\n".join(sentences))
    with open(os.path.join(TARGET_DIR, filename + '.json'), 'w', encoding='utf8') as json_file:
        json.dump(annotations, json_file, ensure_ascii=False)


def create_text_files(label_blacklist=None):
    file, ext = os.path.splitext(SOURCE_FILE)
    sentences = []
    sentence = ''
    char_count = 0
    label_id = 0
    batch_id = 0
    labels = []
    with open(os.path.join(SOURCE_DIR, SOURCE_FILE), 'r') as f:
        for k, txt in enumerate(f.readlines()):
            if len(txt) <= 1:  # the sentence is over
                sentences.append(sentence)  #
                char_count += len(sentence) + 1 #
                if len(sentences) == BATCH_SIZE:
                    labels = filter_labels(labels, label_blacklist)
                    write_files(sentences, labels, file + '_' + str(batch_id))
                    sentences = []
                    labels = []
                    char_count = 0
                    batch_id += 1

                # initialize the new sentence
                sentence = ''
            else:
                word, _, label = txt.split()
                txt_list = txt.split('\t')
                start = char_count + len(sentence) + 1
                sentence += ' ' + word
                end = char_count + len(sentence)
                if label == "O":
                    continue
                elif label[0:2] == 'B-':  # beginning of label
                    label_obj = [start, end, word, label, label_id]
                    labels.append(label_obj)
                    label_id += 1
                else:  # intermediate label
                    label_obj = labels[-1]
                    label_obj[1] += len(word) + 1
                    label_obj[2] += ' ' + word
                    labels[-1] = label_obj

    sentences.append(sentence)

    labels = filter_labels(labels, label_blacklist)
    write_files(sentences, labels, file + '_' + str(batch_id))


def filter_labels(labels, label_blacklist=None):
    if label_blacklist is None:
        return labels
    new_labels = []
    for label_obj in labels:
        if label_obj[3] not in label_blacklist:
            new_labels.append(label_obj)
    return new_labels


def print_label_stats(skip_intermediate=True):
    label_dict = {}
    with open(os.path.join(SOURCE_DIR, SOURCE_FILE), 'r') as f:
        for k, txt in enumerate(f.readlines()):
            if txt[0] == '#':  # a new sentence begins
                sentence = ''
            elif len(txt) > 1:
                txt_list = txt.split('\t')
                label = txt_list[2]
                if skip_intermediate and label[0:2] == 'I-':
                    continue
                if label in label_dict:
                    label_dict[label] += 1
                else:
                    label_dict[label] = 1
    print(json.dumps(label_dict, sort_keys=True, indent=4))


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-t', '--tag-filter-file', dest='filter-file',type=str, default=None,
                        help='A file the contains the comma-separated tags to be excluded from the converted data')
    args = vars(parser.parse_args())

    filter_list = []
    filter_list_file = args['filter-file']
    if filter_list_file is not None:
        if os.path.isfile(filter_list_file):
            fh = open(filter_list_file, 'r')
            filter_list = [tag.strip() for tag in fh.read().split(',')]
        else:
            print('No such file', filter_list_file, '. Ignoring filter list')

    create_text_files(filter_list)
    