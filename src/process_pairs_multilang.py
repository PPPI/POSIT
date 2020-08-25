import os
import sys

from src.preprocessor.preprocess import tokenize_SO_row
from src.tagger.config import Configuration
from src.tagger.model import CodePoSModel


def process_one_rev(model, target_data, rev, post_id):
    with open(os.path.join(os.path.dirname(target_data), 'SO_Posts', post_id, '%d.html' % rev)) as f:
        html = f.read()
    sents_raw = tokenize_SO_row(html, tag_name='div', all_as_code=True)
    os.makedirs('./results/multilang/paired_posts/%s/' % post_id, exist_ok=True)
    for words_raw in sents_raw:
        if len(words_raw) > 0:
            preds = model.predict(words_raw)
            with open('./results/multilang/paired_posts/%s/%d.txt' % (post_id, rev), 'a') as f:
                f.write(' '.join(['%s+%s+%d' % (w, str(t), int(model.config.lang_to_id[l]))
                                  for w, (t, l) in zip(words_raw, zip(*preds))]))
                f.write(' ')


def process_data(model, target_data):
    with open(target_data) as f:
        lines_and_revs = [l.strip().split(',') for l in f.readlines()][1:]
    for postId, rev in lines_and_revs:
        rev = int(rev)
        process_one_rev(model, target_data, rev - 1, postId)
        process_one_rev(model, target_data, rev, postId)


def main():
    # create instance of config
    config = Configuration()
    config.dir_model = sys.argv[1]
    target_data = sys.argv[2]

    # build model
    model = CodePoSModel(config)
    model.build()
    model.restore_session(config.dir_model)

    if not config.multilang:
        print('This code path assumes a multilang model!', file=sys.stderr)
        exit(1)

    # run model over given data
    process_data(model, target_data)


if __name__ == "__main__":
    main()
