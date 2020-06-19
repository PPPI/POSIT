import gzip
import json

from snorkel.labeling import labeling_function

from ..preprocessor.builtin_lists import *
from ..preprocessor.formal_lang_heuristics import is_URI, is_diff_header, is_email

ABSTAIN = -1


def frequency_labeling_function_factory(language):
    location = './data/frequency_data/%s/frequency_data.json.gz' % language
    with gzip.open(location, 'rb') as f:
        frequency_table = json.loads(f.read())[language]

    @labeling_function()
    def lf_frequency_guess(row):
        """
        Return the most frequent tag of `row' in language `language'.
        :param row: The rowen we wish to tag
        :return: The tag in the language
        """
        try:
            return frequency_table[row]
        except KeyError:
            return ABSTAIN

    return lf_frequency_guess


def frequency_language_factory():
    location = './data/frequency_data/frequency_language_data.json.gz'
    with gzip.open(location, 'rb') as f:
        frequency_table = json.loads(f.read())

    @labeling_function()
    def lf_frequency_lang_guess(row):
        """
        Return the most frequent language of `row'.
        :param row: The rowen we wish to identify the language of
        :return: The guessed language
        """
        try:
            return frequency_table[row['Token']]
        except KeyError:
            return ABSTAIN

    return lf_frequency_lang_guess


@labeling_function()
def lf_builtin_language(row):
    if row['Token'] in javascript_builtins:
        return 'javascript'
    elif row['Token'] in golang_builtins:
        return 'golang'
    elif row['Token'] in php_builtins:
        return 'php'
    elif row['Token'] in python_builtins:
        return 'python'
    elif row['Token'] in ruby_builtins:
        return 'ruby'
    else:
        return ABSTAIN


@labeling_function()
def lf_builtin_tag(row):
    if row['Token'] in javascript_builtins:
        return 'Identifier'
    elif row['Token'] in golang_builtins:
        return 'IDENTIFIER'
    elif row['Token'] in php_builtins:
        return 'identifier'
    elif row['Token'] in python_builtins:
        return 'NAME'
    elif row['Token'] in ruby_builtins:
        return 'function_name'
    else:
        return ABSTAIN


@labeling_function()
def lf_uri_lang(row):
    if is_URI(row['Token']):
        return 'uri'
    else:
        return ABSTAIN


@labeling_function()
def lf_uri_row(row):
    if is_URI(row['Token']):
        return 'uri'
    else:
        return ABSTAIN


@labeling_function()
def lf_diff_lang(row):
    if is_diff_header(row['Token']):
        return 'diff'
    else:
        return ABSTAIN


@labeling_function()
def lf_diff_row(row):
    if is_diff_header(row['Token']):
        return 'diff_header'
    else:
        return ABSTAIN


@labeling_function()
def lf_email_lang(row):
    if is_email(row['Token']):
        return 'email'
    else:
        return ABSTAIN


@labeling_function()
def lf_email_row(row):
    if is_email(row['Token']):
        return 'email'
    else:
        return ABSTAIN
