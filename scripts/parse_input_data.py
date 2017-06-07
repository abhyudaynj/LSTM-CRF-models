import argparse
import importlib

parser_module = "bionlp.preprocess.parser"

def parse_args(argparse):
    parser = argparse.ArgumentParser()
    parser.add_argument("input_dir", help="directory containing the brat .ann files")
    parser.add_argument("output_dir", help="directory into which the parsed output files are to be saved")
    parser.add_argument("parser_class", help="name of the Parser class that is to be used for data parsing and imported from the %s module." % parser_module)
    return parser.parse_args()

if __name__ == "__main__":
    args = parse_args(argparse)
    input_dir = args.input_dir
    output_dir = args.output_dir
    parser_class = args.parser_class

    ParserClass = getattr(importlib.import_module(parser_module), parser_class)
    parser = ParserClass()
    parser.parse_datasets(input_dir, output_dir)
