import fileinput
import sys

from gensim.corpora import Dictionary

from src.tagger.data_utils import UNK, O

if __name__ == '__main__':
    dataset = sys.argv[1]
    with_id = sys.argv[2].lower() == 'true'
    with_k = int(sys.argv[3])
    if with_k > -1:
        dataset += '%d'
    else:
        with_k = 1
    for k in range(with_k):
        dataset_k = (dataset % k) if '%d' in dataset else dataset
        file_location_t = './data/corpora/%s/corpus/train.txt' % dataset_k
        file_location_d = './data/corpora/%s/corpus/dev.txt' % dataset_k
        file_location_e = './data/corpora/%s/corpus/eval.txt' % dataset_k
        words = list()
        tags = list()
        chars = list()
        for file_location in [file_location_t, file_location_d, file_location_e]:
            with fileinput.input(file_location, openhook=fileinput.hook_encoded("utf-8")) as f:
                for line in f:
                    line = line.strip()
                    if len(line) > 0:
                        idx = line.rfind(' ')
                        if idx != -1:
                            word, tag = line[:idx], line[idx + 1:]
                            if with_id:
                                idx2 = word.rfind(' ')
                                if idx != -1:
                                    word, tag = word[:idx2], word[idx2 + 1:]
                                    if tag.isdigit():
                                        word = line
                                        tag = UNK
                                else:
                                    word = line
                                    tag = UNK
                            else:
                                if tag.isdigit():
                                    word = line
                                    tag = UNK
                        else:
                            word = line
                            tag = UNK
                        words.append(word)
                        tags.append(tag)
                        for c in word:
                            chars.append(c)

        w_dct = Dictionary([words, [UNK]])
        t_dct = Dictionary([tags, [O]])
        c_dct = Dictionary([chars])

        w_dct.save('./data/corpora/%s/words.dct' % dataset_k)
        t_dct.save('./data/corpora/%s/tags.dct' % dataset_k)
        c_dct.save('./data/corpora/%s/chars.dct' % dataset_k)
