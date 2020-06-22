from gensim.corpora import Dictionary

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
        return tag_dict.token2id[tag]

    def tag_decoding(idx):
        return tag_dict.id2token[idx]

    return tag_encoding, tag_decoding
