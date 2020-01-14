import os
import random
import sys

from .lucid_reader import get_pattern_paths

if __name__ == '__main__':
    file_loc = sys.argv[1]
    with_l_id = sys.argv[2].lower() == 'true'
    offset = 0.3
    train_size = 0.2
    dev_size = 0.1
    eval_size = 0.4
    text = ''
    pattern = '*.c'
    if with_l_id:
        pattern += '_l_id'
    for k in range(20):
        pattern += '%d.txt' % k
        for lucid_file in get_pattern_paths(pattern, file_loc):
            with open(lucid_file, encoding='utf-8') as f:
                text += '\n\n' + f.read()[:-1]
        examples = text.split('\n\n')
        random.seed(42)  # Control seed for reproducibility and debug
        random.shuffle(examples)
        train, dev, evaluation = examples[int(offset * len(examples)):int((offset + train_size) * len(examples))], \
                                 examples[int((offset + train_size) * len(examples)):int(
                                     (offset + train_size + dev_size) * len(examples))], \
                                 examples[int((offset + train_size + dev_size) * len(examples)):int(
                                     (offset + train_size + dev_size + eval_size) * len(examples))]

        os.makedirs('./data/corpora/lucid%s/corpus' % (('_Id%d' if with_l_id else '%d') % k), exist_ok=True)
        with open('./data/corpora/lucid%s/corpus/train.txt' % (('_Id%d' if with_l_id else '%d') % k), 'w',
                  encoding='utf-8') as f:
            f.write('\n\n'.join(train))
        with open('./data/corpora/lucid%s/corpus/dev.txt' % (('_Id%d' if with_l_id else '%d') % k), 'w',
                  encoding='utf-8') as f:
            f.write('\n\n'.join(dev))
        with open('./data/corpora/lucid%s/corpus/eval.txt' % (('_Id%d' if with_l_id else '%d') % k), 'w',
                  encoding='utf-8') as f:
            f.write('\n\n'.join(evaluation))
