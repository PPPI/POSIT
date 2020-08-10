import os
import sys

import pandas as pd
from nltk import pos_tag, word_tokenize
from tqdm import tqdm

from src.preprocessor.codeSearch_preprocessor import languages, formal_languages
from src.snorkel.encoding import lang_decoding, uri_decoding, diff_decoding, email_decoding
from src.snorkel.weak_labellers import *
from src.tagger.data_utils import O, UNK

tag_decoders = {
    **{
        lang: tag_encoding_factory(lang)[1] for lang in [
            'go',
            'javascript',
            'php',
            'python',
            'ruby',
            'java',
        ]
    },
    **{
        'uri': lambda x: uri_decoding[x] if x > 0 and x in uri_decoding.keys() else O,
        'diff': lambda x: diff_decoding[x] if x > 0 and x in uri_decoding.keys() else O,
        'email': lambda x: email_decoding[x] if x > 0 and x in uri_decoding.keys() else O,
    }
}


def main(argv):
    location = argv[0]
    df_train = pd.read_csv(location, index_col=0)

    max_post_id = df_train.iloc[-1]['PostIdx']
    valid_index = int(0.6 * max_post_id)
    test_index = int(0.8 * max_post_id)
    os.makedirs('./data/corpora/multilingual/so', exist_ok=True)
    current_context = ''
    os.makedirs('./data/corpora/multilingual/so/corpus', exist_ok=True)
    for filename in ['eval.txt', 'dev.txt', 'train.txt']:
        with open('./data/corpora/multilingual/so/corpus/%s' % filename, 'w') as f:
            pass
    for index, row in tqdm(df_train.iterrows(), desc='Output'):
        if row['PostIdx'] > max_post_id:
            break

        if row['PostIdx'] > test_index:
            filename = 'eval.txt'
        elif row['PostIdx'] > valid_index:
            filename = 'dev.txt'
        else:
            filename = 'train.txt'
        if row['Context'] != current_context:
            with open('./data/corpora/multilingual/so/corpus/%s' % filename, 'a') as f:
                f.write('\n')
            current_context = row['Context']

        # We use the NLTK pos_tag function for POS tags rather than snorkeling.
        eng_tag = UNK
        for tok, tag in pos_tag(word_tokenize(row['Context'])):
            if tok == str(row['Token']):
                eng_tag = tag
                break

        with open('./data/corpora/multilingual/so/corpus/%s' % filename, 'a') as f:
            to_output = [str(row['Token']), eng_tag] \
                        + [tag_decoders[language](row['label_%s' % language])
                           if row['label_%s' % language] != 0 else O
                           for language in languages + formal_languages] \
                        + [lang_decoding(row['lang_label']) if row['lang_label'] != 0 else row['Language']]
            f.write(' '.join(to_output) + '\n')


if __name__ == '__main__':
    main(sys.argv[1:])
