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


def lang_encoding(lang):
    return lang_encoding_lookup[lang]


def tag_encoding_factory(language):
    tag_dict = Dictionary.load('./data/frequency_data/%s/tags.dct' % language)

    def tag_encoding(tag):
        return tag_dict.doc2idx([tag], unknown_word_index=-1)[0]

    return  tag_encoding
