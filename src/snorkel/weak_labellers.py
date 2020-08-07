import gzip
import json
import re
from collections import defaultdict

from Levenshtein import distance as levenshtein
from snorkel.labeling import labeling_function

from .encoding import lang_encoding, tag_encoding_factory, uri_encoding
from ..preprocessor.builtin_lists import *
from ..preprocessor.codeSearch_preprocessor import UNDEF
from ..preprocessor.formal_lang_heuristics import is_URI, is_diff_header, is_email
from .brute_parse import RowLabeller

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


def frequency_labeling_function_levenshtein_factory(language):
    location = './data/frequency_data/%s/frequency_data.json.gz' % language
    with gzip.open(location, 'rb') as f:
        frequency_table = json.loads(f.read())[language]

    frequency_keys = list(frequency_table.keys())

    def levenstein_warpper(levenshtein_distance=0):
        @labeling_function(name='lf_frequency_lang_guess_%d' % levenshtein_distance)
        def lf_frequency_guess(row):
            """
            Return the most frequent tag of `row' in language `language'.
            :param row: The rowen we wish to tag
            :return: The tag in the language
            """
            candidate_keys = [k for k in frequency_keys if levenshtein(k, row['Token']) <= levenshtein_distance]
            try:
                if len(candidate_keys) > 0:
                    tags = defaultdict(int)
                    for candidate in candidate_keys:
                        for t, fq in frequency_table[candidate].items():
                            if t != UNDEF:
                                tags[t] += fq
                    tags = list(tags.items())
                    tags = sorted(tags, key=lambda p: p[-1], reverse=True)
                    return tag_encoders[language](tags[0])
                else:
                    return ABSTAIN
            except IndexError:
                return ABSTAIN

        return lf_frequency_guess

    return levenstein_warpper


def frequency_language_levenshtein_factory():
    location = './data/frequency_data/frequency_language_data.json.gz'
    with gzip.open(location, 'rb') as f:
        frequency_table = json.loads(f.read())

    frequency_keys = list(frequency_table.keys())

    def levenshtein_wrapper(levenshtein_distance=0):
        @labeling_function(name='lf_frequency_lang_guess_%d' % levenshtein_distance)
        def lf_frequency_lang_guess(row):
            """
            Return the most frequent language of `row'.
            :param row: The rowen we wish to identify the language of
            :return: The guessed language
            """
            candidate_keys = [k for k in frequency_keys if levenshtein(k, row['Token']) <= levenshtein_distance]
            if len(candidate_keys) > 0:
                lang_list = defaultdict(int)
                for candidate in candidate_keys:
                    for l, fq in frequency_table[candidate].items():
                        lang_list[l] += fq
                lang_list = list(lang_list.items())
                lang_list = sorted(lang_list, key=lambda p: p[-1], reverse=True)

                if lang_list[0][-1] > 0:
                    return lang_encoding(lang_list[0][0])
                else:
                    return ABSTAIN
            else:
                return ABSTAIN

        return lf_frequency_lang_guess

    return levenshtein_wrapper


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
def lf_user_language(row):
    language = row['Language'].lower() if row['Language'] != 'English' else 'English'
    return lang_encoding(language)


@labeling_function()
def lf_builtin_tag_factory(language):
    if language == 'javascript':
        @labeling_function()
        def lf_builtin_tag(row):
            if str(row['Token']) in javascript_builtins:
                return tag_encoders['javascript']('IdentifierName')
            else:
                return ABSTAIN
    elif language == 'go':
        @labeling_function()
        def lf_builtin_tag(row):
            if str(row['Token']) in golang_builtins:
                return tag_encoders['go']('IdentifierList')
            else:
                return ABSTAIN

    elif language == 'php':
        @labeling_function()
        def lf_builtin_tag(row):
            if str(row['Token']) in php_builtins:
                return tag_encoders['php']('Identifier')
            else:
                return ABSTAIN

    elif language == 'python':
        @labeling_function()
        def lf_builtin_tag(row):
            if str(row['Token']) in python_builtins:
                return tag_encoders['python']('Expr')
            else:
                return ABSTAIN

    elif language == 'ruby':
        @labeling_function()
        def lf_builtin_tag(row):
            if str(row['Token']) in ruby_builtins:
                return tag_encoders['ruby']('Function_definition')
            else:
                return ABSTAIN

    else:
        lf_builtin_tag = None

    return lf_builtin_tag


def lf_bruteforce_tag_factory(language, tag_encoders):
    rl = RowLabeller()
    if language == 'javascript':
        @labeling_function()
        def lf_bruteforce_tag(row):
            return rl.lookUpToken('javascript', row, tag_encoders)
    elif language == 'go':
        @labeling_function()
        def lf_bruteforce_tag(row):
            return rl.lookUpToken('go', row, tag_encoders)
    elif language == 'php':
        @labeling_function()
        def lf_bruteforce_tag(row):
            return rl.lookUpToken('php', row, tag_encoders)
    elif language == 'python':
        @labeling_function()
        def lf_bruteforce_tag(row):
            return rl.lookUpToken('python', row, tag_encoders)
    elif language == 'ruby':
        @labeling_function()
        def lf_bruteforce_tag(row):
            return rl.lookUpToken('ruby', row, tag_encoders)
    elif language == 'java':
        @labeling_function()
        def lf_bruteforce_tag(row):
            return rl.lookUpToken('java', row, tag_encoders)
    else:
        lf_bruteforce_tag = None

    return lf_bruteforce_tag


@labeling_function()
def lf_formal_lang(row):
    if is_URI(str(row['Token'])):
        return lang_encoding('uri')
    elif is_diff_header(str(row['Token'])):
        return lang_encoding('diff')
    elif is_email(str(row['Token'])):
        return lang_encoding('email')
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
def lf_diff_tok(row):
    if is_diff_header(str(row['Token'])):
        return 1  # There is only a 'diff_header' option
    else:
        return ABSTAIN


@labeling_function()
def lf_email_tok(row):
    if is_email(str(row['Token'])):
        return 1  # THere is only an 'email' option
    else:
        return ABSTAIN
