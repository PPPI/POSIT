import sys

import numpy as np

from src.preprocessor.preprocess import tokenise_SO, tokenise_lkml
from src.tagger.config import Configuration
from src.tagger.model import CodePoSModel

PLs_trained_on = ['go', 'java', 'javascript', 'php', 'python', 'ruby']


def most_frequent(lst):
    return max(set(lst), key=lst.count)


def process_data(model, target_data, stackoverflow=False):
    if stackoverflow:
        source = tokenise_SO(target_data, 75000, 76000, True)
    else:
        source = tokenise_lkml(target_data)
    lid_hits = list()
    for sents_raw, tags in source:
        all_lids = list()
        for words_raw in sents_raw:
            if len(words_raw) > 0:
                preds = model.predict(words_raw)
                all_lids += preds[-1]
                with open(f"./results/multilang/for_manual_investigation_{stackoverflow}.txt", 'a') as f:
                    f.write('\n'.join(['%s\t%s\t%s' % (w, str(t), l) for w, (t, l) in zip(words_raw, zip(*preds))]))
        all_lids = [lid for lid in all_lids if lid != 'English']
        lid_pred = most_frequent(all_lids) if len(all_lids) > 0 else ''
        if any([t in tags for t in PLs_trained_on]):
            lid_hits.append(1 if lid_pred in tags else 0)
        with open(f"./results/multilang/for_manual_investigation_{stackoverflow}.txt", 'a') as f:
            f.write('\n\n' + ''.join(['_'] * 80) + '\n\n')
    print(f"For posts over trained languages, we guessed correctly the user tag in {np.mean(lid_hits):2.3f} cases.")


def main():
    # create instance of config
    config = Configuration()
    config.dir_model = sys.argv[1]
    target_data = sys.argv[2]
    stackoverflow = sys.argv[3].lower() == 'true'

    # build model
    model = CodePoSModel(config)
    model.build()
    model.restore_session(config.dir_model)
    if not config.multilang:
        print('This code path assumes a multilang model!', file=sys.stderr)
        exit(1)

    # run model over given data
    process_data(model, target_data, stackoverflow=stackoverflow)


if __name__ == "__main__":
    main()
