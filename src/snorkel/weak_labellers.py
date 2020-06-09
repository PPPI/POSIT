import gzip
import json

from snorkel.labeling import labeling_function

from ..preprocessor.builtin_lists import *
from ..preprocessor.formal_lang_heuristics import is_URI, is_diff_header, is_email

ABSTAIN = -1


def frequency_labelling_function_factory(language):
    location = './data/frequency_data/frequency_data.json.gz' % language
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


def frequency_language_factory():
    location = './data/frequency_data/frequency_language_data.json.gz'
    with gzip.open(location, 'rb') as f:
        frequency_table = json.loads(f.read())

    @labeling_function
    def lf_frequency_lang_guess(tok):
        """
        Return the most frequent language of `tok'.
        :param tok: The token we wish to identify the language of
        :return: The guessed language
        """
        try:
            return frequency_table[tok]
        except KeyError:
            return ABSTAIN


@labeling_function
def lf_builtin_language(tok):
    if tok in javascript_builtins:
        return 'javascript'
    elif tok in golang_builtins:
        return 'golang'
    elif tok in php_builtins:
        return 'php'
    elif tok in python_builtins:
        return 'python'
    elif tok in ruby_builtins:
        return 'ruby'
    else:
        return ABSTAIN


@labeling_function
def lf_builtin_tag(tok):
    if tok in javascript_builtins:
        return 'Identifier'
    elif tok in golang_builtins:
        return 'IDENTIFIER'
    elif tok in php_builtins:
        return 'identifier'
    elif tok in python_builtins:
        return 'NAME'
    elif tok in ruby_builtins:
        return 'function_name'
    else:
        return ABSTAIN


@labeling_function
def lf_uri_lang(tok):
    if is_URI(tok):
        return 'uri'
    else:
        return ABSTAIN


@labeling_function
def lf_uri_tok(tok):
    if is_URI(tok):
        return 'uri'
    else:
        return ABSTAIN


@labeling_function
def lf_diff_lang(tok):
    if is_diff_header(tok):
        return 'diff'
    else:
        return ABSTAIN


@labeling_function
def lf_diff_tok(tok):
    if is_diff_header(tok):
        return 'diff_header'
    else:
        return ABSTAIN


@labeling_function
def lf_email_lang(tok):
    if is_email(tok):
        return 'email'
    else:
        return ABSTAIN


@labeling_function
def lf_email_tok(tok):
    if is_email(tok):
        return 'email'
    else:
        return ABSTAIN
