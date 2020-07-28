import gzip
import json
import re

from snorkel.labeling import labeling_function

from .encoding import lang_encoding, tag_encoding_factory, uri_encoding
from ..preprocessor.builtin_lists import *
from ..preprocessor.codeSearch_preprocessor import UNDEF
from ..preprocessor.formal_lang_heuristics import is_URI, is_diff_header, is_email
from .brute_parse import BruteParse

ABSTAIN = 0

tag_encoders = {
    lang: tag_encoding_factory(lang)[0] for lang in [
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
            tags = [(t, fq) for t, fq in frequency_table[str(row['Token'])].items() if t != UNDEF]
            sorted(tags, key=lambda p: p[-1], reverse=True)
            return tag_encoders[language](tags[0])
        except (KeyError, IndexError):
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
            lang_list = list(frequency_table[str(row['Token'])].items())
            sorted(lang_list, key=lambda p: p[-1], reverse=True)

            if lang_list[0][-1] > 0:
                return lang_encoding(lang_list[0][1])
            else:
                return ABSTAIN
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
def lf_builtin_tag_factory(language):
    if language == 'javascript':
        @labeling_function()
        def lf_builtin_tag(row):
            bp = BruteParse()
            bp.parse('javascript', row['Context'])
            if str(row['Token']) in javascript_builtins:
                return tag_encoders['javascript']('IdentifierName')
            else:
                return ABSTAIN
    elif language == 'go':
        @labeling_function()
        def lf_builtin_tag(row):
            bp = BruteParse()
            bp.parse('go', row['Context'])
            if str(row['Token']) in golang_builtins:
                return tag_encoders['go']('IdentifierList')
            else:
                return ABSTAIN

    elif language == 'php':
        @labeling_function()
        def lf_builtin_tag(row):
            bp = BruteParse()
            bp.parse('php', row['Context'])
            if str(row['Token']) in php_builtins:
                return tag_encoders['php']('Identifier')
            else:
                return ABSTAIN

    elif language == 'python':
        @labeling_function()
        def lf_builtin_tag(row):
            bp = BruteParse()
            bp.parse('python', row['Context'])
            if str(row['Token']) in python_builtins:
                return tag_encoders['python']('Expr')
            else:
                return ABSTAIN

    elif language == 'ruby':
        @labeling_function()
        def lf_builtin_tag(row):
            bp = BruteParse()
            bp.parse('ruby', row['Context'])
            if str(row['Token']) in ruby_builtins:
                return tag_encoders['ruby']('Function_definition')
            else:
                return ABSTAIN

    else:
        lf_builtin_tag = None

    return lf_builtin_tag


@labeling_function()
def lf_uri_lang(row):
    if is_URI(str(row['Token'])):
        return lang_encoding('uri')
    else:
        return ABSTAIN


@labeling_function()
def lf_uri_tok(row):
    if is_URI(str(row['Token'])):
        tok = str(row['Token'])
        if tok.startswith('file'):
            return uri_encoding['file']
        elif tok.startswith('http'):
            return uri_encoding['http']
        elif tok.startswith('ftp'):
            return uri_encoding['ftp']
        elif tok.startswith('localhost'):
            return uri_encoding['localhost']
        elif re.match(r"\d{1,3}\.\d{1,3}\.\d{1,3}\.\d{1,3}", tok) is not None:
            return uri_encoding['ipv4']
        else:
            return uri_encoding['uri']
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
    if is_diff_header(str(row['Token'])):
        return 1  # There is only a 'diff_header' option
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
    if is_email(str(row['Token'])):
        return 1  # THere is only an 'email' option
    else:
        return ABSTAIN
