import fileinput
import sys

from gensim.corpora import Dictionary

from src.tagger.data_utils import UNK, O

if __name__ == '__main__':
    dataset = sys.argv[1]
    file_location_t = './data/corpora/%s/corpus/train.txt' % dataset
    file_location_d = './data/corpora/%s/corpus/dev.txt' % dataset
    file_location_e = './data/corpora/%s/corpus/eval.txt' % dataset
    words = list()
    tags = list()
    chars = list()
    for file_location in [file_location_t, file_location_d, file_location_e]:
        with fileinput.input(file_location, openhook=fileinput.hook_encoded("utf-8")) as f:
            for line in f:
                line = line.strip()
                if len(line) > 0:
                    parse = line.split(' ')
                    tags_ = parse[-11:]
                    word = ' '.join(parse[:-11])
                    idx = line.rfind(' ')
                    words.append(word)
                    tags += tags_
                    for c in word:
                        chars.append(c)

        w_dct = Dictionary([words, [UNK]])
        t_dct = Dictionary([tags, [O]])
        c_dct = Dictionary([chars])

        w_dct.save('./data/corpora/%s/words.dct' % dataset)
        t_dct.save('./data/corpora/%s/tags.dct' % dataset)
        c_dct.save('./data/corpora/%s/chars.dct' % dataset)
