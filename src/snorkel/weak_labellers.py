from snorkel.labeling import labeling_function


@labeling_function()
def lf_frequency_guess(tok, language):
    """
    Return the most frequent tag of `tok' in language `language'.
    :param tok: The token we wish to tag
    :param language: The language of the tag
    :return: The tag in the language
    """
    pass
