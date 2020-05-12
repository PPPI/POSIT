import gzip
import json

from snorkel.labeling import labeling_function

ABSTAIN = -1


def frequency_labelling_function_factory(language):
    location = './data/frequency_data/%s/frequency_data.json.gz' % language
    with gzip.open(location, 'rb') as f:
        frequency_table = json.loads(f.read())[language]

    @labeling_function()
    def lf_frequency_guess(tok):
        """
        Return the most frequent tag of `tok' in language `language'.
        :param tok: The token we wish to tag
        :return: The tag in the language
        """
        try:
            return frequency_table[tok]
        except KeyError:
            return ABSTAIN

    return lf_frequency_guess
