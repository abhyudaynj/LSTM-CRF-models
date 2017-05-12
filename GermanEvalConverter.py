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
    char_count = 0
    labelId = 0
    annotations = []
    doBreak = False
    with open(os.path.join(SOURCE_DIR, SOURCE_FILE), 'r') as f:
        for k, txt in enumerate(f.readlines()):
            if txt[0] == '#':  # a new sentence begins
                sentence = ''
            elif len(txt) > 1:
                txt_list = txt.split('\t')
                word = txt_list[1]

                if word == '"':  # ignore quotation marks
                    continue

                # if word in ['.', ',', ':', ';', '!', '?']:
                #     sentence += word
                # else:
                start = char_count + len(sentence) + 1
                # end = start + len(word)
                sentence += ' ' + word
                end = char_count + len(sentence)
                if word == 'Köpernicker':
                    print(k, word)
                    doBreak = True;

                label = txt_list[2]
                if label != "O":
                    annotations.append([start, end, word, label, labelId])
                    labelId += 1
            else:  # the sentence is over
                sentences.append(sentence)  #
                char_count += len(sentence) + 1 #

            # if len(sentences) > 200:
            #     file = 'test'
            #     break
            if doBreak:
                sentences.append(sentence)  #
                char_count += len(sentence) + 1  #
                file = 'test'
                break

    sentences.append(sentence)  #
    char_count += len(sentence) + 1  #
    with open(os.path.join(TARGET_DIR, file), 'w') as txt_file:
        txt_file.write("\n".join(sentences))

    with open(os.path.join(TARGET_DIR, file + '.json'), 'w', encoding='utf8') as json_file:
        json.dump(annotations, json_file, ensure_ascii=False)

if __name__ == '__main__':
    create_text_file()
