import pickle, sys, random,logging,os,json
from operator import itemgetter
logger=logging.getLogger(__name__)
IGNORE_TAG='None'

def prepare_document_report(o,l,p,encoded_documents,output_dir):
    logger.info('Preparing the documents reports with the IGNORE_TAG = \'{0}\' ( This is case sensitive)'.format(IGNORE_TAG))
    untouched_tokens=sum([len(sent_token) for sent_token in l])
    logger.info('Total Tokens {0}'.format(untouched_tokens))
    doc_list={}
    for s_id,(sentence,sent_token) in enumerate(o):
        if len(p[s_id]) ==0:
            continue
        for t_id,token in enumerate(sent_token):
            tem=p[s_id]
            if t_id >= len(p[s_id]):
                print t_id, len(p[s_id])
            token.attr['predicted']=tem[t_id]
        produced_sentence=[(tk.value,(tk.attr['offset'],tk.attr['offset']+tk.attr['length']),tk.attr['predicted']) for tk in sent_token]
        if sent_token[0].attr['document'] in doc_list:
            doc_list[sent_token[0].attr['document']].append(label_aggregator(produced_sentence))
        else:
            doc_list[sent_token[0].attr['document']]=[label_aggregator(produced_sentence)]
    logger.info('Writing the predicted annotations to files in {0}'.format(output_dir))
    for idxs,document in enumerate(encoded_documents.value):
        doc_id=document.id
        if doc_id not in doc_list:
            logger.warning('Could not find the Document {0} in the processed output. Please verify'.format(doc_id))
            continue
        doc_text=document.attr['raw_text']
        filename='-'.join(subname for subname in str(doc_id).split('/') if subname.strip()!='')
        with open(os.path.join(output_dir,'{0}.json'.format(filename)),'w') as fout:
            ann_id=0
            json_list=[]
            for res in doc_list[doc_id]:
                for tok in res:
                    if tok[2]!=IGNORE_TAG:
                        json_list.append({'id':ann_id,'type':tok[2],'begin':tok[1][0],'end':tok[1][1],'text':tok[0],'raw_text':doc_text[tok[1][0]:tok[1][1]]})
                        ann_id+=1
            json_obj={}
            json_obj['file_id']=doc_id
            json_obj['predictions']=json_list
            json.dump(json_list,fout)
        with open(os.path.join(output_dir,'{0}.txt'.format(filename)),'w') as fout:
            fout.write(doc_text)
            fout.write('\n')

def label_aggregator(sent):
    idx =0
    while idx < len(sent)-1:
        if sent[idx][2] == sent[idx+1][2] and  sent[idx][2]!=IGNORE_TAG :
            sent[idx:idx+2]=[(sent[idx][0]+' '+sent[idx+1][0],(sent[idx][1][0],sent[idx+1][1][1]),sent[idx][2])]
        else:
            idx+=1
    return sent

