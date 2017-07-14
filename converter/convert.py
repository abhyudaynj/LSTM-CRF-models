import argparse
import os
import utils


def parse_args(parser):
    parser.add_argument('converter', type=str, help='The name of the converter to use for the input data')
    parser.add_argument('input_file', type=str, help='The input file onto which the converter is used')
    parser.add_argument('output_dir', type=str, help='The directory into which to save the converted output')
    parser.add_argument('-b', '--batch_size', type=int, default=1000,
                        help='Maximum amount of sentences per output file. If 0 is given, only one '
                             'output file is created. Default 1000.')
    parser.add_argument('-t', '--tag_filter_file', type=str, default=None,
                        help='A file the contains the comma-separated tags to be excluded from the converted data')
    parser.add_argument('-n', '--ner-mode', action='store_true',
                        help='If set, labels will not be joined to form sequences, but all "B-" and "I-" labels '
                             'will be used (classic NER tagging)')
    return parser.parse_args()


def retrieve_filters(tag_filter_file):
    filters = []
    if tag_filter_file is not None:
        if os.path.isfile(tag_filter_file):
            with open(tag_filter_file, 'r') as fh:
                filters = [tag.strip() for tag in fh.read().split(',')]
        else:
            print("No such file '%s'. Ignoring filter list" % tag_filter_file)
    return filters


def create_conll2000_text_files(input_file, output_dir, batch_size, label_blacklist=None):
    filename, ext = os.path.splitext(os.path.basename(input_file))
    sentences = []
    sentence = ''
    char_count = 0
    label_id = 0
    batch_id = 0
    labels = []
    with open(input_file, 'r') as f:
        for k, txt in enumerate(f.readlines()):
            if len(txt) <= 1:  # the sentence is over
                sentences.append(sentence)  #
                char_count += len(sentence) + 1 #
                if batch_size > 0 and len(sentences) == batch_size:
                    labels = utils.filter_labels(labels, label_blacklist)
                    output_filepath = os.path.join(output_dir, filename + '_' + str(batch_id))
                    utils.write_files(sentences, labels, output_filepath)
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

    labels = utils.filter_labels(labels, label_blacklist)
    output_filepath = os.path.join(output_dir, filename + '_' + str(batch_id))
    utils.write_files(sentences, labels, output_filepath)


def create_germeval_text_files(input_file, output_dir, batch_size, label_blacklist=None, join_labels=True):
    filename, ext = os.path.splitext(os.path.basename(input_file))
    sentences = []
    sentence = ''
    char_count = 0
    label_id = 0
    batch_id = 0
    labels = []
    with open(input_file, 'r') as f:
        for k, txt in enumerate(f.readlines()):
            if txt[0] == '#':  # a new sentence begins
                sentence = ''
            elif len(txt) > 1:
                txt_list = txt.split('\t')
                word = txt_list[1]
                start = char_count + len(sentence) + 1
                sentence += ' ' + word
                end = char_count + len(sentence)
                label = txt_list[2]
                if label == "O":
                    continue
                else:
                    if label[0:2] == 'B-' or not join_labels:
                        # either beginning of label or labels should not be joined into sequences:
                        #  append the label and increment label_id
                        labels.append([start, end, word, label, label_id])
                        label_id += 1
                    else:  # intermediate label: append word and indices to last label_obj
                        labels[-1][2] += ' ' + word
                        labels[-1][1] += 1 + len(word)
            else:  # the sentence is over
                sentences.append(sentence)  #
                char_count += len(sentence) + 1 #
                if batch_size > 0 and len(sentences) == batch_size:
                    labels = utils.filter_labels(labels, label_blacklist)
                    output_filepath = os.path.join(output_dir, filename + '_' + str(batch_id))
                    utils.write_files(sentences, labels, output_filepath)
                    sentences = []
                    labels = []
                    char_count = 0
                    batch_id += 1

    sentences.append(sentence)

    labels = utils.filter_labels(labels, label_blacklist)
    output_filepath = os.path.join(output_dir, filename + '_' + str(batch_id))
    utils.write_files(sentences, labels, output_filepath)


def retrieve_create_text_function(converter):
    if converter.lower() == "germeval":
        return create_germeval_text_files
    elif converter.lower() == "conll":
        return create_conll2000_text_files
    else:
        raise ValueError("'%s' is not a valid converter argument" % converter)


if __name__ == '__main__':
    args = parse_args(argparse.ArgumentParser())
    converter = args.converter
    input_file = args.input_file
    output_dir = args.output_dir
    batch_size = args.batch_size
    tag_filter_file = args.tag_filter_file
    join_labels = not args.ner_mode

    filters = retrieve_filters(tag_filter_file)
    create_text_files = retrieve_create_text_function(converter)
    create_text_files(input_file, output_dir, batch_size, filters, join_labels)