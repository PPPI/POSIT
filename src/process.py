import sys

from .preprocessor.preprocess import tokenise_SO, tokenise_lkml
from .tagger.config import Configuration
from .tagger.model import CodePoSModel


def process_data(model, target_data, stackoverflow=False):
    if stackoverflow:
        source = tokenise_SO(target_data, 75000, 76000)
    else:
        source = tokenise_lkml(target_data)
    for sents_raw in source:
        for words_raw in sents_raw:
            if len(words_raw) > 0:
                preds = model.predict(words_raw)
                if isinstance(preds, tuple):
                    with open('./results/for_manual_investigation.txt', 'a') as f:
                        f.write(' '.join(['%s+%s+%d' % (w, t, l) for w, (t, l) in zip(words_raw, zip(*preds))]))
                        f.write(' ')
                else:
                    with open('./results/for_manual_investigation.txt', 'a') as f:
                        f.write(' '.join(['%s+%s' % (w, t) for w, t in zip(words_raw, preds)]))
                        f.write(' ')
        with open('./results/for_manual_investigation.txt', 'a') as f:
            f.write('\n\n' + ''.join(['_'] * 80) + '\n\n')


def main():
    # create instance of config
    config = Configuration()
    config.dir_model = sys.argv[1]
    target_data = sys.argv[2]

    # build model
    model = CodePoSModel(config)
    model.build()
    model.restore_session(config.dir_model)

    # run model over given data
    process_data(model, target_data)


if __name__ == "__main__":
    main()
