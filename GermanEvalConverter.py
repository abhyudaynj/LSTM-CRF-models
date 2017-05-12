#
# convert the data proided in https://sites.google.com/site/germeval2014ner/ to the requirements of
#  https://github.com/architrave-de/LSTM-CRF-models
#

import os

SOURCE_DIR = "data/sources/GermEval2014_complete_data"
TARGET_DIR = "data/converted/GermEval2014_complete_data"
SOURCE_FILE = "NER-de-train.tsv"


def create_text_file():
    file, ext = os.path.splitext(SOURCE_FILE)
    sentences = []
    sentence = ''
    with open(os.path.join(SOURCE_DIR, SOURCE_FILE), 'r') as f:
        for k, txt in enumerate(f.readlines()):
            if txt[0] == '#':  # a new sentence begins
                sentence = ''
            elif len(txt) == 1:  # the sentence is over
                sentences.append(sentence.strip())
            else:
                txt_list = txt.split('\t')
                word = txt_list[1]
                if word == '"':  # ignore quotation signz
                    continue
                sentence += ' ' + word if word not in ['.', ','] else word
            if k == 52:
                nf = open(os.path.join(TARGET_DIR, file + '.txt'), 'w')
                nf.write("\n".join(sentences))
                nf.close()
                break

if __name__ == '__main__':
    create_text_file()
