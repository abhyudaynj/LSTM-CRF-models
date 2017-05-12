#
# convert the data proided in https://sites.google.com/site/germeval2014ner/ to the requirements of
#  https://github.com/architrave-de/LSTM-CRF-models
#

import os
import json

SOURCE_DIR = "data/sources/GermEval2014_complete_data"
TARGET_DIR = "data/converted/GermEval2014_complete_data/datasets"
SOURCE_FILE = "NER-de-train.tsv"


def create_text_file():
    file, ext = os.path.splitext(SOURCE_FILE)
    sentences = []
    sentence = ''
    labelId = 0
    annotations = []
    with open(os.path.join(SOURCE_DIR, SOURCE_FILE), 'r') as f:
        for k, txt in enumerate(f.readlines()):
            if txt[0] == '#':  # a new sentence begins
                sentence = ''
            elif len(txt) == 1:  # the sentence is over
                sentences.append(sentence.strip())
            else:
                txt_list = txt.split('\t')
                word = txt_list[1]
                label = txt_list[2]
                if word == '"':  # ignore quotation signz
                    continue
                if label != "O":
                    start = len(sentence) + 1
                    end = len(word)
                    annotations.append([start, end, word, label, labelId])
                    labelId += 1
                if word in ['.', ',', ':', ';', '!', '?']:
                    sentence += word
                # elif word in ['(', '„']:
                #     sentence += word
                else:
                    sentence += ' ' + word

    with open(os.path.join(TARGET_DIR, file + '.txt'), 'w') as txt_file:
        txt_file.write("\n".join(sentences))

    with open(os.path.join(TARGET_DIR, file + '.json'), 'w') as json_file:
        json.dump(annotations, json_file)

if __name__ == '__main__':
    create_text_file()
