#
# convert the data proided in https://sites.google.com/site/germeval2014ner/ to the requirements of
#  https://github.com/architrave-de/LSTM-CRF-models
#

import os

SOURCE_DIR = "data/sources/GermEval2014_complete_data"
TARGET_DIR = "data/target/GermEval2014_complete_data"

SOURCE_FILE = "NER-de-train.tsv"


def create_text_file():
    sentences = []
    sentence = ''
    with open(os.path.join(SOURCE_DIR, SOURCE_FILE), 'r') as f:
        content = f.readlines()
        # content_block = content.strip().split('#')
        # print(content_block)
        for k, txt in enumerate(content):
            if len(txt) == 1:
                sentences.append(sentence)
            elif txt[0] == '#':
                sentence = ''
            else:
                txt_list = txt.strip().split('\t')
                word = txt_list[1]
                sentence += word + ' '
            if k == 50:
                print(sentences)
                break




if __name__ == '__main__':
    create_text_file()
