from gensim.corpora import Dictionary

from src.tagger.data_utils import O

lang_encoding_lookup = {
    'English': 1,
    'go': 2,
    'javascript': 3,
    'php': 4,
    'python': 5,
    'ruby': 6,
    'java': 7,
    'uri': 8,
    'email': 9,
    'diff': 10,
}

uri_encoding = {
    'file': 1,
    'http': 2,
    'ftp': 3,
    'localhost': 4,
    'ipv4': 5,
    'uri': 6,
}

uri_decoding = {v: k for k, v in uri_encoding.items()}

diff_encoding = {'diff_header': 1}
diff_decoding = {1: 'diff_header'}

email_encoding = {'email': 1}
email_decoding = {1: 'email'}

lang_decode_lookup = {v: k for k, v in lang_encoding_lookup.items()}


def lang_encoding(lang):
    return lang_encoding_lookup[lang]


def lang_decoding(lang):
    return lang_decode_lookup[lang]


def tag_encoding_factory(language):
    tag_dict = Dictionary.load('./data/frequency_data/%s/tags.dct' % language)

    def tag_encoding(tag):
        try:
            return tag_dict.token2id[tag] + 1
        except KeyError:
            tag_dict.add_documents([[tag]])
            tag_dict.save('./data/frequency_data/%s/tags.dct' % language)
            return tag_dict.token2id[tag] + 1

    def tag_decoding(idx):
        if idx == 0:
            return O
        else:
            return tag_dict[idx - 1]

    return tag_encoding, tag_decoding
