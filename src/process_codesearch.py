import re
import sys

import json_lines as jl
import numpy as np
from nltk import sent_tokenize
from tqdm import tqdm

from src.preprocessor.codeSearch_preprocessor import location_format
from src.preprocessor.util import CODE_TOKENISATION_REGEX
from src.tagger.config import Configuration
from src.tagger.model import CodePoSModel

PLs_trained_on = ['java', 'javascript', 'php', 'python', 'ruby', 'go']
# (language, fold, language, fold, fold number)
jsonl_location_format = '%s\\final\\jsonl\\%s\\%s_%s_%d_parsed.jsonl.gz'

def load_and_process(model, language):
    for fold in tqdm(['test'], leave=False, desc="Fold"):
        # Determine number of files, we error fast, but don't actually read the file by using jl.open()
        n_files = 0
        while True:
            try:
                location = (location_format + jsonl_location_format) \
                           % (language, language, fold, language, fold, n_files)
                with jl.open(location) as _:
                    pass
            except FileNotFoundError:
                break
            finally:
                n_files += 1

        for i in tqdm(range(n_files - 1), leave=False, desc='Files'):
            location = (location_format + jsonl_location_format) % (language, language, fold, language, fold, i)
            process_with_posit(model, location, language)


def most_frequent(lst):
    return max(set(lst), key=lst.count)


def least_frequent(lst):
    return min(set(lst), key=lst.count)


def load_source(location):
    with jl.open(location) as f:
        for entry in f:
            toks_ = [
                [l.strip()
                 for l in re.findall(CODE_TOKENISATION_REGEX,
                                     line.strip())
                 if len(l.strip()) > 0]
                for line in sent_tokenize(entry['docstring_parsed'])
            ]
            yield toks_


def process_with_posit(model, location, language):
    source = load_source(location)

    lid_hits = list()
    lid_hits_at_3 = list()

    # Clear file before run
    with open(f"./results/multilang/for_manual_investigation_codesearch_{language}.txt", 'w') as _:
        pass

    for sents_raw in source:
        all_lids = list()
        for words_raw in sents_raw:
            if len(words_raw) > 0:
                preds = model.predict(words_raw)
                all_lids += preds[-1]
                with open(f"./results/multilang/for_manual_investigation_codesearch_{language}.txt", 'a') as f:
                    f.write('\n'.join(['%s\t%s\t%s' % (w, str(t), l) for w, (t, l) in zip(words_raw, zip(*preds))]))
                    f.write('\n')
        all_lids = [lid for lid in all_lids if lid != 'English']
        lid_pred = most_frequent(all_lids) if len(all_lids) > 0 else ''

        while len(set(all_lids)) > 3:
            exclude = least_frequent(all_lids)
            all_lids = [lid for lid in all_lids if lid != exclude]

        top_3 = set(all_lids)

        if any([t in [language] for t in PLs_trained_on]) and len(lid_pred) > 0:
            lid_hits.append(1 if lid_pred in [language] else 0)
            lid_hits_at_3.append(1 if any([lid in [language] for lid in top_3]) else 0)
        with open(f"./results/multilang/for_manual_investigation_codesearch_{language}.txt", 'a') as f:
            f.write('\n\n' + ''.join(['_'] * 80) + '\n\n')
    print(f'Language: {language}')
    print(f"For posts over trained languages, we guessed correctly the user tag in {np.mean(lid_hits):2.3f} cases.")
    print(f"For posts over trained languages, we guessed correctly in top 3 "
          f"the user tag in {np.mean(lid_hits):2.3f} cases.")


def main():
    # create instance of config
    config = Configuration()
    config.dir_model = sys.argv[1]

    # build model
    model = CodePoSModel(config)
    model.build()
    model.restore_session(config.dir_model)
    if not config.multilang:
        print('This code path assumes a multilang model!', file=sys.stderr)
        exit(1)

    # run model over given data
    for language in PLs_trained_on:
        load_and_process(model, language)


if __name__ == "__main__":
    main()
