from __future__ import division
from tqdm import tqdm
from nltk import word_tokenize,sent_tokenize
import re,json,os
import logging
import nltk
import pickle

logging.basicConfig(level=logging.INFO)
logger=logging.getLogger(__name__)

from string import punctuation

PNT=[p for p in punctuation]
SPNT=PNT+[' ']

USEFULL=',.'
USELIST=[p for p in USEFULL]

USELESS =[p for p in PNT if p not in USELIST]
USELESS=[]


mtch=0.0
mismt=0.0

def verify_positions(anns,txt):
    valid_matches=[]
    global mtch
    global mismt
    for ann in anns:
        anno=txt[ann[0]:ann[1]]
        ptxt=ann[2]
        anno=str(''.join(anno.split("\\n")))
        ptxt=str(''.join(ptxt.split("\\n")))
        if ''.join(re.split("[\r\n\s]",anno)) !=''.join(re.split("[\r\n\s]",ptxt)):
            print "Annotation id {0}".format(ann[4])
            print "mismatch{2} \'{0}\'\n ------------- instead of ------ \n\'{1}\'\n ------- found at provided position \n\n".format(''.join(re.split("[\r\n]",anno)),''.join(re.split("[\r\n]",ptxt)),(ann[0],ann[1],ann[3]))
            mismt+=1.0
        else:
            mtch+=1.0
            valid_matches.append(ann)
    return valid_matches

def match_words(s1,s2):
    if len(s1) != len(s2):
        logger.warning('Length of processed sentences are different.')
        return False
    else:
        match_flag=True
        for i,x in enumerate(s1):
            if s1[i][0] != s1[i][0]:
                match_flag=False
        return match_flag

def prepareSents(wrds):
    valid_sents=[]
    text=''.join(wrd[0] for wrd in wrds)
    sent_list=[[(word,0,'None') for word in sent] for sent in sent_tokenize(text)]
    text=[word for word in wrds if word[0]!=' ']
    sent_list=[[word for word in concat_words(strip_chars(sent)) if word[0]!=' '] for sent in sent_list]
    idx=0
    s_idx=0
    while idx < len(text) and s_idx<len(sent_list):
        if not match_words(sent_list[s_idx],text[idx:idx+len(sent_list[s_idx])]):
            print "NLTK:"+ str(sent_list[s_idx])
            print 'MINE:' + str(text[idx:idx+len(sent_list[s_idx])])
        else:
            valid_sents+=[text[idx:idx+len(sent_list[s_idx])]]
        idx=idx+len(sent_list[s_idx])
        s_idx+=1
    return valid_sents

def build_char_annotations(anns,txt):
    chr_list=[(x,chr_idx,'None') for chr_idx,x in enumerate(txt)]
    for start,stop,text,types,ann_ids in anns:
        if stop >= len(chr_list):
            logger.warning('Annotation id {0} is out of bounds of the text provided'.format(ann_ids))
            print('Annotation id {0} is out of bounds of the text provided'.format(ann_ids))
        chr_list[start:stop]=[(charac[0],start+chr_idx,types) for chr_idx,charac in enumerate(chr_list[start:stop])]
    tkn_list=concat_words(strip_chars(chr_list))
    return prepareSents(tkn_list)


def strip_chars(chr_list):

    for i,(char,chr_idx,label) in enumerate(chr_list):
        if char == '\r' or char == '\n':
            chr_list[i]=(' ',chr_idx,label)
    cmpr_str=[]
    chr_list=remove_symbols(chr_list)
    for i,(char,chr_idx,label) in enumerate(chr_list):
        if i>0 and chr_list[i-1][0]==' ' and char[0]==' ':
            continue
        else:
            cmpr_str+=[chr_list[i]]
    idx =0
    return cmpr_str


def concat_words(chr_list):
    idx=0
    while idx<len(chr_list):
        if idx<len(chr_list)-1 and chr_list[idx][0] not in SPNT and chr_list[idx+1][0] not in SPNT:
            chr_list[idx]=(chr_list[idx][0]+chr_list[idx+1][0],chr_list[idx][1],chr_list[idx][2])
            del chr_list[idx+1]
        else:
            idx+=1
    return chr_list


def remove_symbols(chr_list):
    wrds=[]
    for x in chr_list:
        if x[0] not in USELESS:
            wrds+=[x]
        else:
            wrds+=[(' ',x[1],x[2])]
    return wrds

def file_extractor(input_list_file,umls_param):
    # Needs an input file with list of text file locations
    if umls_param!=0:
        logger.info('UMLS parameter is on. I will extract all umls annotations from *.umls.json . If I dont find any such file, I will populate empty umls features')
    raw_text=[]
    notes =[]
    list_file =[ filename for filename in open(input_list_file,'r').readlines() if filename.strip()!='' ]
    for filename in tqdm(list_file):
        metamap_anns=[]
        file_text= open(filename.strip(),'r').read()
        #Checking for existing UMLS files
        if umls_param!=0:
            if os.path.isfile('{0}.umls.json'.format(filename.strip())):
                metamap_anns=json.load(open('{0}.umls.json'.format(filename.strip()),'r'))
            else:
                logger.warning('UMLS file not found for {0}. Populating with empty umls annotations'.format(filename.strip()))
        raw_text.append((filename.strip(),file_text,metamap_anns))
        notes.append((filename.strip(),build_char_annotations([],file_text)))
    return raw_text,notes

def annotated_file_extractor(input_list_file,umls_param):
    '''
    Needs an input file with list of text file locations.
    Each input text file 'filepath/filename' should have a json file at 'filepath/filename.json'.
    The json file should contain a list of annotation objects.
    Each annotation object is itself a list of the following format [start char offset, end char offset, annotated text, annotation type, annotation id]
    '''
    if umls_param!=0:
        logger.info('UMLS parameter is on. I will extract all umls annotations from *.umls.json . If I dont find any such file, I will populate empty umls features')

    raw_text=[]
    notes =[]
    list_file =[ filename for filename in open(input_list_file,'r').readlines() if filename.strip()!='' ]
    for filename in tqdm(list_file):
        metamap_anns=[]
        assert (os.path.isfile(filename.strip()) and os.path.isfile('{0}.json'.format(filename.strip()))), 'The provided filename {0} either does not exist or does not have an annotation file of format .json'.format(filename.strip())
        file_text= open(filename.strip(),'r').read()
        anns=json.load(open('{0}.json'.format(filename.strip()),'r'))
        #Checking for existing UMLS files
        if umls_param!=0:
            if os.path.isfile('{0}.umls.json'.format(filename.strip())):
                metamap_anns=json.load(open('{0}.umls.json'.format(filename.strip()),'r'))
            else:
                logger.warning('UMLS file not found for {0}. Populating with empty umls annotations'.format(filename.strip()))
        valid_anns=verify_positions(anns,file_text)
        notes.append((filename.strip(),build_char_annotations(valid_anns,file_text)))
        raw_text.append((filename.strip(),file_text,metamap_anns))
    return raw_text,notes
