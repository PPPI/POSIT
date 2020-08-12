import csv
import fileinput
import re
import sys
from multiprocessing import Pool

import bs4
import pandas as pd
import psutil
import xmltodict
from bs4 import BeautifulSoup
from nltk import sent_tokenize
from nltk.tokenize import TreebankWordTokenizer as twt
from tqdm import tqdm

from src.preprocessor.util import HTML_PARSER, CODE_TOKENISATION_REGEX

tokenizer = twt()


def parse_stackoverflow_posts(file_location):
    with fileinput.input(file_location, openhook=fileinput.hook_encoded("utf-8")) as f_:
        for line in f_:
            line = line.strip()
            if line.startswith("<row"):
                row_ = xmltodict.parse(line)['row']
                if '@Tags' in row_.keys():
                    if '<java>' in row_['@Tags']:
                        language = 'java'
                    elif '<golang>' in row_['@Tags'] or '<go>' in row_['@Tags']:
                        language = 'golang'
                    elif '<javascript>' in row_['@Tags']:
                        language = 'javascript'
                    elif '<php>' in row_['@Tags']:
                        language = 'php'
                    elif '<ruby>' in row_['@Tags']:
                        language = 'ruby'
                    elif '<python>' in row_['@Tags']:
                        language = 'python'
                    else:
                        language = 'abstain'

                    if language != 'abstain':
                        yield row_['@Body'], language


def tokenize_SO_row(row_, language, tag_name='body'):
    row_ = BeautifulSoup(row_, HTML_PARSER).find(tag_name)
    text__ = [(tag.text, 'Code' if tag.name == 'pre' or tag.name == 'code' else 'NL')
              for tag in row_.childGenerator() if isinstance(tag, bs4.element.Tag)]
    text___ = list()
    for (body_, kind_) in text__:
        try:
            if kind_ == 'NL':
                toks_ = [(s[start:end], 'English', (start, end), body_)
                         for s in sent_tokenize(body_) for (start, end) in tokenizer.span_tokenize(s)]
            elif kind_ == 'Code':
                toks_ = [
                    (l.group(), language, l.span(), body_)
                    for line in body_.split('\n')
                    for l in re.finditer(CODE_TOKENISATION_REGEX,
                                         line.strip())
                    if len(l.group().strip()) > 0
                ]
            text___ += toks_
        except (ValueError, IndexError):
            # Some sentences are malformed, we drop them from training
            pass
    return text___


def tokenize_SO_row_star(args):
    return tokenize_SO_row(*args)


def tokenise_SO(location_, offset_, limit_):
    for location_, (row_, language_) in enumerate(parse_stackoverflow_posts(location_)):
        if location_ < offset_:
            continue
        if location_ >= limit_:
            break
        yield row_, language_


def SO_to_pandas(location, limit=None, offset=None):
    fn_out = location[:-len('.xml')]
    if offset is not None:
        fn_out += '_%d' % offset
    fn_out += '.csv'
    try:
        result_df = pd.read_csv(fn_out)
    except FileNotFoundError:
        number_of_posts_after_filter = 5748669  # Precomputed separately by iterating through the file once.

        if offset is not None:
            number_of_posts_after_filter = max(number_of_posts_after_filter - offset, 0)
            limit += offset
        else:
            offset = 0

        if number_of_posts_after_filter == 0:
            return pd.DataFrame(columns=['PostIdx', 'Token', 'Language', 'Span', 'Context'], data=[])

        if limit is not None:
            number_of_posts_after_filter = min(limit-offset, number_of_posts_after_filter)

        with open(fn_out, 'w', encoding='utf-8') as f:
            f.write(','.join(['PostIdx', 'Token', 'Language', 'Span', 'Context']) + '\n')
        with Pool(processes=len(psutil.Process().cpu_affinity()), maxtasksperchild=4) as wp:
            for pidx, toks in \
                    tqdm(enumerate(wp.imap(tokenize_SO_row_star, tokenise_SO(location, offset, limit))),
                         total=number_of_posts_after_filter):
                for token, lang, span, context in toks:
                    with open(fn_out, 'a', encoding='utf-8') as f:
                        csvwriter = csv.writer(f)
                        csvwriter.writerow([pidx,
                                            token.replace('"', '""'),
                                            lang,
                                            str(span),
                                            context])
                if pidx == limit:
                    break
            result_df = pd.read_csv(fn_out)

    return result_df


if __name__ == '__main__':
    location = sys.argv[1]
    try:
        limit = int(sys.argv[2])
    except (IndexError, ValueError):
        limit = None
    try:
        offset = int(sys.argv[3])
    except (IndexError, ValueError):
        offset = None
    df = SO_to_pandas(location, limit, offset)
