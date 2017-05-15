#
# convert the data proided in https://sites.google.com/site/germeval2014ner/ to the requirements of
#  https://github.com/architrave-de/LSTM-CRF-models
#

import os
import json

SOURCE_DIR = "data/sources/GermEval2014_complete_data"
TARGET_DIR = "data/converted/GermEval2014_complete_data/datasets"
SOURCE_FILE = "NER-de-train.tsv"
BATCH_SIZE = 1000


def write_files(sentences, annotations, filename):
    with open(os.path.join(TARGET_DIR, filename), 'w') as txt_file:
        txt_file.write("\n".join(sentences))
    with open(os.path.join(TARGET_DIR, filename + '.json'), 'w', encoding='utf8') as json_file:
        json.dump(annotations, json_file, ensure_ascii=False)


def create_text_files():
    file, ext = os.path.splitext(SOURCE_FILE)
    sentences = []
    sentence = ''
    char_count = 0
    label_id = 0
    batch_id = 0
    labels = []
    with open(os.path.join(SOURCE_DIR, SOURCE_FILE), 'r') as f:
        for k, txt in enumerate(f.readlines()):
            if txt[0] == '#':  # a new sentence begins
                sentence = ''
            elif len(txt) > 1:
                txt_list = txt.split('\t')
                word = txt_list[1]
                # if word == '"':  # ignore quotation marks
                #     continue

                start = char_count + len(sentence) + 1
                sentence += ' ' + word
                end = char_count + len(sentence)
                label = txt_list[2]
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
            else:  # the sentence is over
                sentences.append(sentence)  #
                char_count += len(sentence) + 1 #
                if len(sentences) == BATCH_SIZE:
                    write_files(sentences, labels, file + '_' + str(batch_id))
                    sentences = []
                    labels = []
                    char_count = 0
                    batch_id += 1

    sentences.append(sentence)  #
    write_files(sentences, labels, file + '_' + str(batch_id))

if __name__ == '__main__':
    create_text_files()