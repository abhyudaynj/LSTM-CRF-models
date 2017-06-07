from abc import ABC, abstractmethod
import json
import os

from bionlp.tags import IOB2_Tag

class AbstractDatasetParser(ABC):
    @abstractmethod
    def parse_datasets(self, input_dir, output_dir):
        raise NotImplementedError

    @abstractmethod
    def parse_file(self, input_file, output_file):
        raise NotImplementedError


class BratParser(AbstractDatasetParser):
    # this is implemented in the BratParser class instead of the abstract parent class because,
    # for Brat, we only want to regard the input files with file extension .ann
    def parse_datasets(self, input_dir, output_dir):
        # this methodology only expects files in the top level of the given input directory, not subdirs
        _, _, input_files = next(os.walk(input_dir))
        for input_file in input_files:
            if input_file.endswith(".ann"):
                input_file_abs_path = os.path.join(input_dir, input_file)
                output_file_abs_path = os.path.join(output_dir, os.path.splitext(input_file)[0] + ".json")
                self.parse_file(input_file_abs_path, output_file_abs_path)

    def parse_file(self, input_file, output_file):
        total_tags_list = []
        with open(input_file, "r", encoding="utf8") as f_in:
            tag_chain_beginning_index = None
            tag_chain_end_index = None
            tag_chain_token_values = ""
            tag_chain_tag = None
            tag_chain_id = -1
            for line in f_in.readlines():
                cleaned_line = line.rstrip()
                brat_index, tag_and_indices, token_value = cleaned_line.split("\t")
                tag_string, indices = self.split_tag_and_indices(tag_and_indices)
                tag = IOB2_Tag(tag_string)
                earliest_start_index, latest_end_index = self.get_earliest_start_and_latest_end_index(indices)
                if IOB2_Tag.is_tag_continuation_of_beginning_tag(tag_chain_tag, tag):
                    tag_chain_end_index = latest_end_index
                    tag_chain_token_values = " ".join([tag_chain_token_values, token_value])
                else:
                    # if a previous tag chain exists, append it to the total tags list
                    if tag_chain_tag is not None:
                        self.append_tag_chain_to_total_list(total_tags_list, tag_chain_beginning_index,
                            tag_chain_end_index, tag_chain_token_values, tag_chain_tag, tag_chain_id)

                    # begin new tag chain
                    tag_chain_beginning_index = earliest_start_index
                    tag_chain_end_index = latest_end_index
                    tag_chain_token_values = token_value
                    tag_chain_tag = tag
                    tag_chain_id += 1

            # after the final line has been read, write the final tag chain into the total tags list
            self.append_tag_chain_to_total_list(total_tags_list, tag_chain_beginning_index,
                            tag_chain_end_index, tag_chain_token_values, tag_chain_tag, tag_chain_id)

        # write the total tags list into the output file
        with open(output_file, "w", encoding="utf8") as f_out:
            json.dump(total_tags_list, f_out, ensure_ascii=False)

    def append_tag_chain_to_total_list(self, total_tags_list, tag_chain_beginning_index, tag_chain_end_index, tag_chain_token_values, tag_chain_tag, tag_chain_id):
        tag_entry = [   tag_chain_beginning_index,
                        tag_chain_end_index,
                        tag_chain_token_values,
                        tag_chain_tag.value,
                        tag_chain_id
                    ]
        total_tags_list.append(tag_entry)

    def split_tag_and_indices(self, tag_and_indices):
        tag, indices = tag_and_indices.split(" ", 1)
        return tag, indices

    def get_earliest_start_and_latest_end_index(self, indices):
        """
        Indices can be denoted in different formats; if only a single token is tagged with a
        certain tag consecutively, the start and end indices of that tag will look as follows:
        "10 15"
        But the same tag can be assigned to multiple consecutive tokens, in which case the indices
        string will look as follows:
        "10 15;16 18;19 25"

        This method will retrieve the earliest start index and the latest end index (10 and 25 in
        the second example above).
        """
        index_tuples = []
        start_and_end_indices = indices.split(";")
        earliest_start_index = int(start_and_end_indices[0].split()[0])
        latest_end_index = int(start_and_end_indices[-1].split()[1])
        return earliest_start_index, latest_end_index
