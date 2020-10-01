import fileinput
import os
import sys

import numpy as np
import xmltodict
from nltk.tokenize import TreebankWordTokenizer as twt
from tqdm import tqdm

tokenizer = twt()


def parse_stackoverflow_posts(file_location, condition):
    condition = set(condition)
    with fileinput.input(file_location, openhook=fileinput.hook_encoded("utf-8")) as f_:
        for line in f_:
            line = line.strip()
            if line.startswith("<row"):
                row_ = xmltodict.parse(line)['row']
                if '@Tags' in row_.keys():
                    if set(row_['@Tags']).intersection(condition) == condition:
                        yield row_['@Body']


def main(args):
    # Fixed for reproducibility
    np.random.seed(42)

    # CLI Args
    location = args[0]
    languages = args[1:]

    posts_overall = {l: list() for l in languages}
    for language in tqdm(languages, desc="Languages"):
        posts_overall[language] = list(parse_stackoverflow_posts(location, ['<%s>' % language]))

    for language in tqdm(languages, desc="Languages"):
        selected_idx = list(np.random.choice(len(posts_overall[language]), 30, replace=False))
        selected = list()
        for idx in selected_idx:
            selected.append(posts_overall[language][idx])

        os.makedirs('./data/corpora/so_conditional/%s' % language, exist_ok=True)
        with open('./data/corpora/so_conditional/%s/sampled_posts.txt' % language, 'w') as f:
            f.write('\n<DOC-END>\n'.join(selected))


if __name__ == '__main__':
    main(sys.argv[1:])
