from tqdm import tqdm
import re
import json
import os
import logging
from string import punctuation

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

PNT = [p for p in punctuation]
SPNT = PNT + [' ']

USEFULL = ',.'
USELIST = [p for p in USEFULL]

USELESS = [p for p in PNT if p not in USELIST]
USELESS = []


mtch = 0.0
mismt = 0.0

# TODO: Remove hardcoded file name
mismatch_log_file_path = "./mismatches.log"


def verify_positions(anns, txt, filename):
    valid_matches = []
    global mtch
    global mismt
    mismt_in_file = 0
    for ann in anns:
        anno = txt[ann[0]:ann[1]]
        ptxt = ann[2]
        anno = str(''.join(anno.split("\\n")))
        ptxt = str(''.join(ptxt.split("\\n")))
        if ''.join(re.split("[\r\n\s]", anno)) != ''.join(re.split("[\r\n\s]", ptxt)):
            # TODO: Include option to toggle output on mismatch
            mismt += 1.0
            mismt_in_file += 1
            # print("Annotation id {0}".format(ann[4]))
            # print("mismatch{2} \'{0}\'\n ------------- instead of ------ \n\'{1}\'\n ------- found at provided position \n\n".format(
            #     ''.join(re.split("[\r\n]", anno)), ''.join(re.split("[\r\n]", ptxt)), (ann[0], ann[1], ann[3])))
            # raise Exception('Preparation error in training data')
        else:
            mtch += 1.0
            valid_matches.append(ann)
    if mismt_in_file > 0:
        with open(mismatch_log_file_path, "a") as f_mismatches:
            f_mismatches.write(filename)
    return valid_matches


def match_words(s1, s2):
    if len(s1) != len(s2):
        logger.warning('Length of processed sentences are different.')
        return False
    else:
        match_flag = True
        for i, x in enumerate(s1):
            if s1[i][0] != s1[i][0]:
                match_flag = False
        return match_flag


def prepareSents(wrds, complete_text, sentence_delim='\n'):
    """
    From a list of tuples that contain information about each token in the text,
    and a string that contains the complete text (including the sentence delimiters),
    create a list of lists for the sentences in the text.

    wrds : list
        A list that contains information about each token in the text, with each element
        in the list being a 3-tuple.
        first element of each 3-tuple: str
            the token itself
        second element of each 3-tuple: int
            the index that the first character of the token has in the total text
        third element of each 3-tuple: str
            the target tag associated with the token

    complete_text : str
        The complete text as one string. The text is expected to have its sentences
        separated by the given sentence delimiter.

    sentence_delim : str
        The string that acts as the sentence delimiter in complete_text.

    returns : list
        Similar to the given argument wrds, a list is returned, but it is now a list
        of lists. Each sublist represents a sentence, and contains the 3-tuples given
        by wrds for its tokens. This function thus acts as a means of splitting the
        tokens given by wrds into their respective sentences.
    """
    valid_sents = []
    sent_list = [[(word, 0, 'None') for word in sent]
                 for sent in complete_text.split(sentence_delim)]
    word_list = [word for word in wrds if word[0] != ' ']
    sent_list = [[word for word in concat_words(
        strip_chars(sent)) if word[0] != ' '] for sent in sent_list]
    idx = 0
    s_idx = 0
    while idx < len(word_list) and s_idx < len(sent_list):
        if not match_words(sent_list[s_idx], word_list[idx:idx + len(sent_list[s_idx])]):
            print("NLTK:" + str(sent_list[s_idx]))
            print('MINE:' + str(word_list[idx:idx + len(sent_list[s_idx])]))
        else:
            valid_sents += [word_list[idx:idx + len(sent_list[s_idx])]]
        idx = idx + len(sent_list[s_idx])
        s_idx += 1
    return valid_sents


def build_char_annotations(anns, txt):
    chr_list = [(x, chr_idx, 'None') for chr_idx, x in enumerate(txt)]
    for start, stop, text, types, ann_ids in anns:
        if stop >= len(chr_list):
            logger.warning(
                'Annotation id {0} is out of bounds of the text provided'.format(ann_ids))
            print(
                ('Annotation id {0} is out of bounds of the text provided'.format(ann_ids)))
        chr_list[start:stop] = [(charac[0], start + chr_idx, types)
                                for chr_idx, charac in enumerate(chr_list[start:stop])]
    tkn_list = concat_words(strip_chars(chr_list))
    return prepareSents(tkn_list, txt)


def strip_chars(chr_list):
    for i, (char, chr_idx, label) in enumerate(chr_list):
        if char == '\r' or char == '\n':
            chr_list[i] = (' ', chr_idx, label)
    cmpr_str = []
    chr_list = remove_symbols(chr_list)
    for i, (char, chr_idx, label) in enumerate(chr_list):
        if i > 0 and chr_list[i - 1][0] == ' ' and char[0] == ' ':
            continue
        else:
            cmpr_str += [chr_list[i]]
    return cmpr_str


def concat_words(chr_list):
    idx = 0
    while idx < len(chr_list):
        if idx < len(chr_list) - 1 and chr_list[idx][0] not in SPNT and chr_list[idx + 1][0] not in SPNT:
            chr_list[idx] = (chr_list[idx][0] + chr_list[idx + 1]
                             [0], chr_list[idx][1], chr_list[idx][2])
            del chr_list[idx + 1]
        else:
            idx += 1
    return chr_list


def remove_symbols(chr_list):
    wrds = []
    for x in chr_list:
        if x[0] not in USELESS:
            wrds += [x]
        else:
            wrds += [(' ', x[1], x[2])]
    return wrds


def get_text_from_files(input_list_file, umls_param, include_annotations=False):
    # Needs an input file with list of text file locations
    #
    # If include_annotations is True, each input text file 'filepath/filename' should have a json file at
    # 'filepath/filename.json'. The json file should contain a list of annotation objects.
    # Each annotation object is itself a list of the following format
    # [start char offset, end char offset, annotated text, annotation type, annotation id]
    if umls_param != 0:
        logger.info('UMLS parameter is on. I will extract all umls annotations from *.umls.json. '
                    'If I dont find any such file, I will populate empty umls features')
    raw_text = []
    notes = []
    list_file = [filename for filename in open(input_list_file, 'r').readlines() if filename.strip() != '']

    # if the file that logs mismatches already exists, remove it first
    if include_annotations and os.path.isfile(mismatch_log_file_path):
        os.remove(mismatch_log_file_path)

    for filename in tqdm(list_file):
        metamap_anns = []
        file_text = open(filename.strip(), 'r').read()
        # Checking for existing UMLS files
        if umls_param != 0:
            if os.path.isfile('{0}.umls.json'.format(filename.strip())):
                metamap_anns = json.load(open('{0}.umls.json'.format(filename.strip()), 'r'))
            else:
                logger.warning('UMLS file not found for {0}. '
                               'Populating with empty umls annotations'.format(filename.strip()))
        valid_anns = []
        if include_annotations:
            assert (os.path.isfile(filename.strip()) and os.path.isfile('{0}.json'.format(filename.strip()))), \
                'The provided filename {0} either does not exist ' \
                'or does not have an annotation file of format .json'.format(filename.strip())
            anns = json.load(open('{0}.json'.format(filename.strip()), 'r'))
            valid_anns = verify_positions(anns, file_text, filename)

        raw_text.append((filename.strip(), file_text, metamap_anns))
        notes.append((filename.strip(), build_char_annotations(valid_anns, file_text)))
    return raw_text, notes
