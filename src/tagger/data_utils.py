# Identifiers used to replace UNK words and numbers. Make sure they don't clash with code terms!
import numpy as np
from gensim.corpora import Dictionary

UNK = "X"  # We want to be consistent with the universal tag-set
NUM = "NUM"  # We want to be consistent with the universal tag-set
O = "@O@"  # We use this to denote code unknown


class CorpusIterator(object):
    """
    Class that reads a SOTorrent format dataset and iterates over it. We use the Standford Standard PoS tagger
    to generate tags for English and a snippet compiler to generate code entity tags.
    """

    def __init__(self, filename, processing_word, processing_tag, with_l_id, max_iter=None, offset_lid=0):
        """
        :param filename: Location of the dataset file
        :param processing_word: A function that converts words to ids
        :param processing_tag: A function that converts tags to ids
        :param with_l_id: If language IDs exist in the data
        :param max_iter: (optional) maximum number of elements to yield
        """
        self.filename = filename
        self.processing_word = processing_word
        self.processing_tag = processing_tag
        self.max_iter = max_iter
        self.with_l_id = with_l_id
        self.offset_lid = offset_lid
        self.length = None

    def __len__(self):
        if self.length is None:
            self.length = 0
            for _ in self:
                self.length += 1

        return self.length

    def __iter__(self):
        niter = 0
        with open(self.filename, encoding='utf-8') as f:
            words, tags, l_ids = [], [], []
            for line in f:
                line = line.strip()
                if len(line) == 0 or line.startswith("-DOCSTART-"):
                    if len(words) != 0:
                        niter += 1
                        if self.max_iter is not None and niter > self.max_iter:
                            break
                        yield words, tags, l_ids
                        words, tags, l_ids = [], [], []
                else:
                    idx = line.rfind(' ')
                    if idx != -1:
                        word, tag = line[:idx], line[idx + 1:]
                        l_id = 0
                        if self.with_l_id:
                            idx2 = word.rfind(' ')
                            if idx != -1:
                                l_id = int(tag)
                                word, tag = word[:idx2], word[idx2 + 1:]
                                if tag.isdigit():
                                    word = line
                                    tag = UNK
                                    l_id = 0
                            else:
                                word = line
                                tag = UNK
                                l_id = 0
                        else:
                            if tag.isdigit():
                                word = line
                                tag = UNK
                                l_id = 0
                    else:
                        word = line
                        tag = UNK
                        l_id = 0
                    if self.processing_word is not None:
                        word = self.processing_word(word)
                    if self.processing_tag is not None:
                        tag = self.processing_tag(tag)
                    words += [word]
                    tags += [tag]
                    l_ids += [l_id + self.offset_lid]


def get_vocabs(datasets):
    """Build vocabulary from an iterable of datasets objects

    :param datasets: a list of dataset objects
    :return A pair of Dictionary (gensim). The first is the words vocabulary and the second is the Tag vocabulary
    """
    print("Building vocab...")
    text_coprus = list()
    tag_corpus = list()
    for dataset in datasets:
        for words, tags in dataset:
            text_coprus.append(words)
            tag_corpus.append(tags)
    vocab_words = Dictionary(text_coprus)
    vocab_tags = Dictionary(tag_corpus)
    print("- done. {} tokens".format(len(vocab_words)))
    return vocab_words, vocab_tags


def get_char_vocab(dataset):
    """Build char vocabulary from a dataset iterator

    :param dataset: a iterator yielding tuples (sentence, tags)
    :return a Dictionary (gensim) of all chars in the dataset
    """
    vocab_char = set()
    for words, _ in dataset:
        for word in words:
            vocab_char.update(word)

    return Dictionary(vocab_char)


def camel(s):
    return s != s.lower() and s != s.upper() and "_" not in s


def snake(s):
    return s == s.lower() and '_' in s


def generate_feature_vector(word):
    features = [
        1 if word.isupper() else 0,
        1 if word.istitle() else 0,
        1 if word.islower() else 0,
        1 if camel(word) else 0,
        1 if snake(word) else 0,
        1 if any(char.isupper() for char in word[1:]) else 0,
        1 if any(char.isdigit() for char in word) else 0,
        1 if set('[~!@#$%^&*()_+{}":;\']+$').intersection(word) else 0,
    ]
    features = np.asarray(features)
    return features


def get_processing_word(vocab_words=None, vocab_chars=None,
                        lowercase=False, chars=False, allow_unk=True, feature_vector=False, offset=0):
    """
    Return lambda function that transform a word (string) into list,
    or tuple of (list, id) of int corresponding to the ids of the word and
    its corresponding characters.

    :param vocab_words: dictionary mapping words to ids
    :param vocab_chars:  dictionary mapping chars to ids
    :param lowercase: True if all words should be converted to lowecase
    :param chars: True if the char level encoding should be used
    :param allow_unk: True if we should use the special entity UNK
    :param feature_vector: True if we should return the feature vector as well
    :param offset: ID offset in case 0 has a special meaning, such as padding
    :return: a callable that returns the encoding of a word.
    """

    def f(word):
        # 0. get chars of words
        if vocab_chars is not None and chars is True:
            chars_ = [c for c in word]
            char_ids = vocab_chars.doc2idx(chars_, unknown_word_index=0)

        # 1. preprocess word
        if lowercase:
            word = word.lower()
        if word.isdigit():
            word = NUM

        # 2. Optionally generate feature vector
        if feature_vector:
            features = generate_feature_vector(word)

        # 3. get id of word
        if vocab_words is not None:
            if word in vocab_words.token2id.keys():
                word = vocab_words.token2id[word] + offset
            else:
                if allow_unk:
                    word = len(vocab_words.token2id) - 1 + offset
                else:
                    raise Exception("Unknown key is not allowed. Check that "
                                    "your vocab (tags?) is correct")

        # 4. return tuple char ids, word id
        if vocab_chars is not None and chars is True:
            if feature_vector:
                return features, char_ids, word
            else:
                return char_ids, word
        else:
            if feature_vector:
                return features, word
            else:
                return word

    return f


def _pad_sequences(sequences, pad_tok, max_length):
    """
    Args:
        sequences: a generator of list or tuple
        pad_tok: the char to pad with

    Returns:
        a list of list where each sublist has same length
    """
    sequence_padded, sequence_length = [], []

    for seq in sequences:
        seq = list(seq)
        seq_ = seq[:max_length] + [pad_tok] * max(max_length - len(seq), 0)
        sequence_padded += [seq_]
        sequence_length += [min(len(seq), max_length)]

    return sequence_padded, sequence_length


def pad_sequences(sequences, pad_tok, nlevels=1):
    """
    Args:
        sequences: a generator of list or tuple
        pad_tok: the char to pad with
        nlevels: "depth" of padding, for the case where we have characters ids

    Returns:
        a list of list where each sublist has same length

    """
    if nlevels == 1:
        max_length = max(map(lambda x: len(list(x)), sequences))
        sequence_padded, sequence_length = _pad_sequences(sequences,
                                                          pad_tok, max_length)

    elif nlevels == 2:
        max_length_word = max([max(map(lambda x: len(x), seq))
                               for seq in sequences])
        sequence_padded, sequence_length = [], []
        for seq in sequences:
            # all words are same length now
            sp, sl = _pad_sequences(seq, pad_tok, max_length_word)
            sequence_padded += [sp]
            sequence_length += [sl]

        max_length_sentence = max(map(lambda x: len(x), sequences))
        sequence_padded, _ = _pad_sequences(sequence_padded,
                                            [pad_tok] * max_length_word, max_length_sentence)
        sequence_length, _ = _pad_sequences(sequence_length, 0,
                                            max_length_sentence)

    return sequence_padded, sequence_length


def minibatches(data, minibatch_size):
    """
    Args:
        data: generator of (sentence, tags) tuples
        minibatch_size: (int)

    Yields:
        list of tuples

    """
    x_batch, y_batch, z_batch = [], [], []
    for (x, y, z) in data:
        if len(x_batch) == minibatch_size:
            yield x_batch, y_batch, z_batch
            x_batch, y_batch, z_batch = [], [], []

        if type(x[0]) == tuple:
            x = zip(*x)
        x_batch += [x]
        y_batch += [y]
        z_batch += [z]

    if len(x_batch) != 0:
        yield x_batch, y_batch, z_batch
