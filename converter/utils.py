import os
import json

def write_files(sentences, annotations, filepath):
    with open(filepath, 'w') as txt_file:
        txt_file.write("\n".join(sentences))
    with open(os.path.join(filepath + '.json'), 'w', encoding='utf8') as json_file:
        json.dump(annotations, json_file, ensure_ascii=False)

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