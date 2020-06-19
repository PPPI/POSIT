import gzip
import json
import re

from snorkel.labeling import labeling_function

from .encoding import lang_encoding, tag_encoding_factory
from ..preprocessor.builtin_lists import *
from ..preprocessor.formal_lang_heuristics import is_URI, is_diff_header, is_email

ABSTAIN = -1

tag_encoders = {
    lang: tag_encoding_factory(lang) for lang in [
        'go',
        'javascript',
        'php',
        'python',
        'ruby',
        'java',
    ]
}


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
            return tag_encoders[language](frequency_table[str(row['Token'])])
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
            return lang_encoding(frequency_table[str(row['Token'])])
        except KeyError:
            return ABSTAIN

    return lf_frequency_lang_guess


@labeling_function()
def lf_builtin_language(row):
    if str(row['Token']) in javascript_builtins:
        return lang_encoding('javascript')
    elif str(row['Token']) in golang_builtins:
        return lang_encoding('go')
    elif str(row['Token']) in php_builtins:
        return lang_encoding('php')
    elif str(row['Token']) in python_builtins:
        return lang_encoding('python')
    elif str(row['Token']) in ruby_builtins:
        return lang_encoding('ruby')
    else:
        return ABSTAIN


@labeling_function()
def lf_builtin_tag(row):
    if str(row['Token']) in javascript_builtins:
        return tag_encoders['javascript']('Identifier')
    elif str(row['Token']) in golang_builtins:
        return tag_encoders['go']('IDENTIFIER')
    elif str(row['Token']) in php_builtins:
        return tag_encoders['php']('identifier')
    elif str(row['Token']) in python_builtins:
        return tag_encoders['python']('NAME')
    elif str(row['Token']) in ruby_builtins:
        return tag_encoders['ruby']('function_name')
    else:
        return ABSTAIN


@labeling_function()
def lf_uri_lang(row):
    if is_URI(str(row['Token'])):
        return lang_encoding('uri')
    else:
        return ABSTAIN


@labeling_function()
def lf_uri_tok(row):
    # TODO: Encoding
    if is_URI(str(row['Token'])):
        tok = str(row['Token'])
        if tok.startswith('file'):
            return 'file'
        elif tok.startswith('http'):
            return 'http'
        elif tok.startswith('ftp'):
            return 'ftp'
        elif tok.startswith('localhost'):
            return 'localhost'
        elif re.match(r"\d{1,3}\.\d{1,3}\.\d{1,3}\.\d{1,3}", tok) is not None:
            return 'ipv4'
        else:
            return 'uri'
    else:
        return ABSTAIN


@labeling_function()
def lf_diff_lang(row):
    if is_diff_header(str(row['Token'])):
        return lang_encoding('diff')
    else:
        return ABSTAIN


@labeling_function()
def lf_diff_tok(row):
    # TODO: Encoding
    if is_diff_header(str(row['Token'])):
        return 'diff_header'
    else:
        return ABSTAIN


@labeling_function()
def lf_email_lang(row):
    if is_email(str(row['Token'])):
        return lang_encoding('email')
    else:
        return ABSTAIN


@labeling_function()
def lf_email_tok(row):
    # TODO: Encoding
    if is_email(str(row['Token'])):
        return 'email'
    else:
        return ABSTAIN
