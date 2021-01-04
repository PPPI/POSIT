import fileinput
import json
import re
import sys

import bs4
import xmltodict
from bs4 import BeautifulSoup
from nltk import pos_tag, casual_tokenize
from nltk.tokenize import sent_tokenize

from src.preprocessor.util import operators, line_comment_start, _UNIVERSAL_TAGS, keywords, CODE_TOKENISATION_REGEX, \
    HTML_PARSER
from src.tagger.data_utils import O, snake, camel


def parse_stackoverflow_posts(file_location, for_stormed_=False, return_tags=False):
    with fileinput.input(file_location, openhook=fileinput.hook_encoded("utf-8")) as f_:
        for line in f_:
            line = line.strip()
            if line.startswith("<row "):
                row_ = xmltodict.parse(line)['row']
                if for_stormed_ and ('@Tags' not in row_.keys() or '<java>' not in row_['@Tags']):
                    continue
                if return_tags:
                    if '@Tags' not in row_.keys():
                        continue
                    else:
                        yield row_['@Body'], row_['@Tags']
                else:
                    yield row_['@Body']


def heuristic_tag(tok, pos, tokens):
    if tok in keywords:
        return 'keyword'
    elif tok in operators or (tok[-1] == ';' and tok[:-1] in operators):
        return 'op'
    elif pos + 1 < len(tokens) and '(' in tokens[pos + 1]:
        return 'method'
    elif pos + 1 < len(tokens) and (tokens[pos + 1] in operators
                                    or (tokens[pos + 1][-1] == ';' and tokens[pos + 1][:-1] in operators)):
        return 'variable'
    elif re.match(r"\"[^\"]*\"|'[^']*'", tok):
        return 'string_literal'
    else:
        return O


def annotate_comment_line(tokens):
    return [(tokens[0], O)] + pos_tag(tokens[1:], tagset="universal")


def annotate_line(tokens, context_=None):
    if len(tokens) > 0 and tokens[0] in line_comment_start:
        return annotate_comment_line(tokens), context_
    context_ = dict() if context_ is None else context_
    result = list()
    for pos, tok in enumerate(tokens):
        tok = tok.strip()
        if '.' in tok and not (re.match(r"\"[^\"]*\"|'[^']*'", tok)):
            tok_ = tok.split('.')[-1]
        else:
            tok_ = tok

        tag = heuristic_tag(tok_, pos, tokens)
        if tag in ['method', 'variable']:
            context_[tok_] = tag
        elif tag == O:
            if tok_ in context_.keys():
                tag = context_[tok_]
            elif snake(tok_) or camel(tok_):
                tag = 'variable'
        result.append((tok, tag))
    return result, context_


def annotate_line_using_only_context(tokens, context_=None, freq_context_=None):
    if freq_context_ is None:
        raise ValueError("context cannot be None")
    result = list()
    for pos, tok in enumerate(tokens):
        if '.' in tok and not (re.match(r"\"[^\"]*\"|'[^']*'", tok)):
            tok_ = tok.split('.')[-1]
        else:
            tok_ = tok

        if tok_ in freq_context_.keys():
            try:
                tag = sorted(filter(lambda x: x[0] not in _UNIVERSAL_TAGS, freq_context_[tok_].items()),
                             reverse=True, key=lambda p: p[1])[0][0]
            except IndexError:
                tag = heuristic_tag(tok_, pos, tokens)
        else:
            tag = heuristic_tag(tok_, pos, tokens)

        if tag not in [O, 'keyword', 'op']:
            context_[tok_] = tag
        elif tag == O:
            if tok_ in context_.keys():
                tag = context_[tok_]
            elif snake(tok_) or camel(tok_):
                tag = 'variable'

        result.append((tok, tag))

    return result, context_


def code_tag(snippet, context_=None, context_only=True, freq_context_=None, casual=False):
    if casual:
        tokenised = [casual_tokenize(s) for s in snippet.split('\n')]
    else:
        tokenised = [
            [l.strip()
             for l in re.findall(CODE_TOKENISATION_REGEX,
                                 line.strip())
             if len(l.strip()) > 0]
            for line in snippet.split('\n')
        ]
    result = list()
    context_ = dict() if context_ is None else context_
    if context_only:
        assert freq_context_ is not None
        for tokens in tokenised:
            tagged, context_ = annotate_line_using_only_context(tokens, context_, freq_context_)
            result.append(tagged)
    else:
        for tokens in tokenised:
            tagged, context_ = annotate_line(tokens, context_)
            result.append(tagged)
    return result, context_


def tokenize_SO_row(row_, tag_name='body', all_as_code=False):
    row_ = BeautifulSoup(row_, HTML_PARSER).find(tag_name)
    text__ = [(tag.text, 'Code' if tag.name == 'pre' or tag.name == 'code' else 'NL')
              for tag in row_.childGenerator() if isinstance(tag, bs4.element.Tag)]
    text___ = list()
    for (body_, kind_) in text__:
        if kind_ == 'NL' and not all_as_code:
            toks_ = [casual_tokenize(s) for s in sent_tokenize(body_)]
        elif all_as_code or kind_ == 'Code':
            toks_ = [
                [l.strip()
                 for l in re.findall(CODE_TOKENISATION_REGEX,
                                     line.strip())
                 if len(l.strip()) > 0]
                for line in body_.split('\n')
            ]
        text___ += toks_
    return text___


def tokenise_SO(location_, offset_, limit_, return_tags=False):
    for location_, row_ in enumerate(parse_stackoverflow_posts(location_, return_tags=return_tags)):
        if location_ < offset_:
            continue
        if location_ >= limit_:
            break
        if return_tags:
            row_, tags = row_
            yield tokenize_SO_row(row_), tags[1:-1].split('><')
        else:
            yield tokenize_SO_row(row_)


def remove_leading_symbols(line):
    while len(line) > 0 and set('+-@>').intersection(line[0]):
        line = line[1:].strip()
    return line


def tokenise_lkml(location_):
    with open(location_) as f_:
        text__ = f_.read()
    text__ = text__.split('\n\n' + ''.join(['_'] * 80) + '\n\n')
    for mail in text__:
        mail = '\n'.join([remove_leading_symbols(line) for line in mail.split('\n')])
        toks_ = [
            [l.strip()
             for l in re.findall(CODE_TOKENISATION_REGEX,
                                 line.strip())
             if len(l.strip()) > 0]
            for line in sent_tokenize(mail)
        ]
        yield toks_


if __name__ == '__main__':
    # Iterate through data and generate the vocabulary objects
    # For English, the following works:
    # '\n\n'.join(['\n'.join(['%s %s' % w for w in pos_tag(word_tokenize(s))]) for s in sent_tokenize(row)])
    location = sys.argv[1]
    output_name = sys.argv[2]
    offset = int(sys.argv[3])
    limit = int(sys.argv[4])
    frequency = sys.argv[5].lower() == 'true'
    language_id = sys.argv[6].lower() == 'true'
    try:
        for_stormed = sys.argv[7].lower() == 'true'
    except IndexError:
        for_stormed = False
    limit += offset
    # Prep file, make sure we clear it first
    with open('./data/corpora/SO%s/%s.txt'
              % (('_Freq' if frequency else '') + ('_Id' if language_id else ''), output_name),
              'w', encoding='utf-8') as f:
        pass

    try:
        if frequency:
            with open('./data/corpora/SO/frequency_map.json', encoding='utf-8') as f:
                freq_context = json.loads(f.read())
        else:
            freq_context = None
    except FileNotFoundError:
        freq_context = None

    for location, row in enumerate(parse_stackoverflow_posts(location, for_stormed_=for_stormed)):
        if location < offset:
            continue
        if location >= limit:
            break
        row = BeautifulSoup(row, HTML_PARSER).find('body')
        text = [(tag.text, 'Code' if tag.name == 'pre' or tag.name == 'code' else 'NL')
                for tag in row.childGenerator() if isinstance(tag, bs4.element.Tag)]
        text_ = list()
        context = dict()
        for i, (body, kind) in enumerate(text):
            if kind == 'Code':
                toks, context = code_tag(body, context, frequency, freq_context)
                if language_id:
                    toks = [[(tok, tag, 1) for tok, tag in s] for s in toks]
                text_.append((i, toks))
        for i, (body, kind) in enumerate(text):
            if kind == 'NL':
                toks = [pos_tag(casual_tokenize(s), tagset="universal") for s in sent_tokenize(body)]
                toks = [[
                    tuple([w, t if t not in ['NOUN', 'VERB'] or w not in context.keys() else context[w]] +
                          ([0 if t not in ['NOUN', 'VERB'] or w not in context.keys() else 1]
                           if language_id else []))
                    for w, t in s] for s in toks]
                text_.append((i, toks))
        text_ = [t for _, t in sorted(text_, key=lambda p: p[0])]
        text = [[[t for t in s] for s in p if len(s) > 0] for p in text_]
        formatted_output = ''.join(['\n'.join([
            '%s %s %d' % t if language_id else '%s %s' % t for t in s]) + '\n\n' for p in text for s in p])
        with open('./data/corpora/SO%s/%s.txt'
                  % (('_Freq' if frequency else '') + ('_Id' if language_id else ''), output_name),
                  'a', encoding='utf-8') as f:
            f.write(formatted_output)
