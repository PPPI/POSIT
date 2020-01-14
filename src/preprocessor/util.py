from .reserved_keywords import python_reserved, java_reserved, javascript_reserved, c_reserved, cpp_reserved

HTML_PARSER = "html5lib"
operators = ('=', '/', '+', '*', '^', '%', '&', '&&', '::', '++', '--', '-',
             '->', '!', '~', '>>', '<<', '<', '<=', '>', '>=', '==',
             '!=', '|', '||', '?', ':', ',', '+=', '-=', '*=', '/=', '%=',
             '<<=', '>>=', '&=', '^=', '|=', ';', '(', ')', 'new', 'return', 'yield')
line_comment_start = ('//', '/*', '%', '#',)
# Note that we exclude the '.' tag as we reuse it in code as well.
_UNIVERSAL_TAGS = ('VERB', 'NOUN', 'PRON', 'ADJ', 'ADV', 'ADP', 'CONJ', 'DET', 'NUM', 'PRT', 'X')

keywords = set(python_reserved + java_reserved + javascript_reserved + c_reserved + cpp_reserved)

# The first group is to avoid splitting marked code tokens
CODE_TOKENISATION_REGEX = r"<tt>.*</tt>|" \
                          r"[.;:]|" \
                          r"[^\"'A-Za-z0-9_$.\-\s]{1,3}|" \
                          r"[A-Za-z\-]+n't|[A-Za-z\-]+'d|" \
                          r"[A-Za-z\-]+'s|" \
                          r"_{0,2}[A-Za-z0-9_$.]+\b(?:\(\))?|" \
                          r"\"(?:\\\"|" \
                          r"[^\"])*\"|" \
                          r"'(?:\\'|[^'])*'"
