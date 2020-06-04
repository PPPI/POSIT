import fileinput
import re
import sys

import bs4
import pandas as pd
import xmltodict
from bs4 import BeautifulSoup
from nltk import casual_tokenize, sent_tokenize

from src.preprocessor.util import HTML_PARSER, CODE_TOKENISATION_REGEX


def parse_stackoverflow_posts(file_location):
    with fileinput.input(file_location, openhook=fileinput.hook_encoded("utf-8")) as f_:
        for line in f_:
            line = line.strip()
            if line.startswith("<row_"):
                row_ = xmltodict.parse(line)['row']
                if '@Tags' in row_.keys():
                    if '<java>' in row_['@Tags']:
                        language = 'java'
                    elif '<golang>' in row_['@Tags'] or '<go>' in row_['@Tags']:
                        language = 'go'
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
        if kind_ == 'NL':
            toks_ = [(t, 'English', idx, body_)
                     for idx, t in enumerate(casual_tokenize(s)) for s in sent_tokenize(body_)]
        elif kind_ == 'Code':
            toks_ = [
                (l.strip(), language, idx, body_)
                for idx, l in enumerate(re.findall(CODE_TOKENISATION_REGEX,
                                                   line.strip()))
                if len(l.strip()) > 0
                for line in body_.split('\n')
            ]
        text___ += toks_
    return text___


def SO_to_pandas(location):
    result_df = pd.DataFrame(columns=['Token', 'Language', 'Index', 'Context'])
    for row, language in parse_stackoverflow_posts(location):
        toks = tokenize_SO_row(row, language)
        temp_df = pd.DataFrame(toks, columns=['Token', 'Language', 'Index', 'Context'])
        result_df = result_df.append(temp_df, ignore_index=True, sort=False)

    return result_df


if __name__ == '__main__':
    location = sys.argv[1]
    df = SO_to_pandas(location)
