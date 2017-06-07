from abc import ABC

class AbstractTag(ABC):
    def __init__(self, value):
        self.value = value

class IOB2_Tag(AbstractTag):
    """
    Class for tags following the IOB2 (Inside Outside Beginning) format.
    This class is specifically made for handling Inside and Beginning tags, i.e.
    all tags except for the simple "O" tag.
    """
    def get_tag_prefix(self):
        """
        Returns the tag prefix, which is expected to either be "B-" or "I-".
        """
        return self.value[:2]

    def get_tag_type(self):
        """
        Returns the tag type, which is expected to be whatever follows either
        a "B-" or a "I-" prefix.
        """
        return self.value[2:]

    def is_begin_tag(self):
        """
        Returns True if the tag starts with "B-"; False otherwise.
        """
        return self.get_tag_prefix() == "B-"

    def is_inside_tag(self):
        """
        Returns True if the tag starts with "I-"; False otherwise.
        """
        return self.get_tag_prefix() == "I-"

    @staticmethod
    def is_tag_continuation_of_beginning_tag(beginning_tag, current_tag):
        """
        Determine if the given current tag continues a multiword tag started
        by the given beginning tag.

        Example where the current tag continues the multiword tag:
            beginning_tag: B-Per
            current_tag:   I-Per

        Examples where the current tag DOES NOT continue the multiword tag:
            beginning_tag: B-Per
            current_tag:   B-Loc

            beginning_tag: B-Per
            current_tag:   B-Per

        Basically: the current_tag only counts as a continuation if it starts
        with "I-" and is of the same tag type as the beginning tag.
        """
        return  isinstance(beginning_tag, IOB2_Tag) and \
                isinstance(current_tag, IOB2_Tag) and \
                current_tag.is_inside_tag() and \
                beginning_tag.get_tag_type() == current_tag.get_tag_type()