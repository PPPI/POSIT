import gzip
import json
from collections import deque

# Used to parse code to ASTs
import antlr4
# Enables reading the corpus
import json_lines as jl
from nltk import sent_tokenize, casual_tokenize, pos_tag

# Individual languages that we want to parse
from src.antlr4_language_parsers.golang.GoLexer import GoLexer as gol
from src.antlr4_language_parsers.golang.GoParser import GoParser as gop
from src.antlr4_language_parsers.java.Java9Lexer import Java9Lexer as javal
from src.antlr4_language_parsers.java.Java9Parser import Java9Parser as javap
from src.antlr4_language_parsers.javadoc.JavadocLexer import JavadocLexer as javadocl
from src.antlr4_language_parsers.javadoc.JavadocParser import JavadocParser as javadocp
from src.antlr4_language_parsers.javascript.ECMAScriptLexer import ECMAScriptLexer as jsl
from src.antlr4_language_parsers.javascript.ECMAScriptParser import ECMAScriptParser as jsp
from src.antlr4_language_parsers.php.PhpLexer import PhpLexer as phpl
from src.antlr4_language_parsers.php.PhpParser import PhpParser as phpp
from src.antlr4_language_parsers.python.Python3Lexer import Python3Lexer as pyl
from src.antlr4_language_parsers.python.Python3Parser import Python3Parser as pyp
from src.antlr4_language_parsers.ruby.CorundumLexer import CorundumLexer as rubyl
from src.antlr4_language_parsers.ruby.CorundumParser import CorundumParser as rubyp

# Configuration
languages = [
    'go',
    'java',
    'javascript',
    'php',
    'python',
    'ruby',
]

natural_languages = [
    'English',
]

# language
location_format = 'H:\\CodeSearch\\%s\\'
# language
pickles = [
    '%s_dedupe_definitions_v2.pkl',
    '%s_licenses.pkl',
]

folds = [
    'train',
    'test',
    'validation',
]
# (language, fold, language, fold, fold number)
jsonl_location_format = '%s\\final\\jsonl\\%s\\%s_%s_%d.jsonl.gz'

UNDEF = 'UNDEF'


def parse_java(entry):
    code = entry['code']
    lexer = javal(antlr4.InputStream(code))
    stream = antlr4.CommonTokenStream(lexer)
    parser = javap(stream)
    tree = parser.methodDeclaration()
    return tree


def parse_javadoc(entry):
    docstring = entry['docstring']
    lexer = javadocl(antlr4.InputStream(docstring))
    stream = antlr4.CommonTokenStream(lexer)
    parser = javadocp(stream)
    tree = parser.documentation()
    return tree


def parse_golang(entry):
    code = entry['code']
    lexer = gol(antlr4.InputStream(code))
    stream = antlr4.CommonTokenStream(lexer)
    parser = gop(stream)
    tree = parser.methodDecl()
    return tree


def parse_js(entry):
    code = entry['code']
    lexer = jsl(antlr4.InputStream(code))
    stream = antlr4.CommonTokenStream(lexer)
    parser = jsp(stream)
    tree = parser.program()
    return tree


def parse_php(entry):
    code = entry['code']
    lexer = phpl(antlr4.InputStream(code))
    stream = antlr4.CommonTokenStream(lexer)
    parser = phpp(stream)
    tree = parser.functionDeclaration()
    return tree


def parse_python(entry):
    code = entry['code']
    lexer = pyl(antlr4.InputStream(code))
    stream = antlr4.CommonTokenStream(lexer)
    parser = pyp(stream)
    tree = parser.file_input()
    return tree


def parse_ruby(entry):
    code = entry['code']
    lexer = rubyl(antlr4.InputStream(code))
    stream = antlr4.CommonTokenStream(lexer)
    parser = rubyp(stream)
    tree = parser.prog()
    return tree


def extract_name_from_terminal(terminal_node):
    return str(type(terminal_node.parentCtx)).split('.')[-1][:-9]


def ast_to_tagged_list(tree):
    result = list()
    visit = deque([tree])

    while len(visit) > 0:
        current = visit.pop()
        if current.children is not None:
            for child in current.children:
                if isinstance(child, antlr4.tree.Tree.TerminalNodeImpl):
                    tag = extract_name_from_terminal(child)
                    result.append((child.symbol.text, tag))
                else:
                    visit.append(child)

    return [l for l in reversed(result)]


def java_doc_string_to_nltk(tagged_list):
    output = ''
    add_be_after_next = False
    for text, tag in tagged_list:
        if tag == 'DescriptionLineNoSpaceNoAt':
            if text == '.':
                output = output[:-1]
            output += text + ' '
        elif tag == 'BlockTagTextElement' or tag == 'BlockTagContent':
            if tag == 'BlockTagContent' and output[-1] != '.':
                output += '.'
            output += text
            if tag == 'BlockTagTextElement' and add_be_after_next:
                output += ' is'
                add_be_after_next = False
        elif tag == 'BlockTagName':
            if text == 'param':
                add_be_after_next = True

    return output


def parse_docstring(entry, language, code_context):
    result = list()
    if language == 'java':
        docstring_parse = parse_javadoc(entry)
        tagged_list = ast_to_tagged_list(docstring_parse)
        docstring = java_doc_string_to_nltk(tagged_list)
    else:
        docstring = entry['docstring']

    tagged_sents = [pos_tag(casual_tokenize(s)) for s in sent_tokenize(docstring)]
    for sent in tagged_sents:
        for tok, tag_ in sent:
            if tok in code_context.keys():
                context = code_context[tok]
                result.append(
                    (tok, ('English', {k: context[-1][k] if k != 'English' else tag_ for k in context[-1].keys()}))
                )
            else:
                result.append(
                    (tok, ('English', {l: tag_ if l == 'English' else UNDEF for l in languages + natural_languages})))

    return result


def preprocess_corpus_file(location, language):
    preprocessed_data = ''
    with jl.open(location) as f:
        for entry in f:
            ast = globals()["parse_%s" % language](entry)
            tagged_code_list = [
                (tok, (language, {l: tag if l == language else UNDEF for l in languages + natural_languages}))
                for tok, tag in ast_to_tagged_list(ast)
            ]
            tagged_docstring_list = parse_docstring(entry, language, dict(tagged_code_list))
            entry['code_parsed'] = json.dumps(tagged_code_list)
            entry['docstring_parsed'] = json.dumps(tagged_docstring_list)
            preprocessed_data += json.dumps(entry) + '\n'

    with gzip.open(location[:-len('.jsonl.gz')] + '_parsed.jsonl.gz', 'wb') as f:
        f.write(preprocessed_data)


def main():
    for language in languages:
        for fold in folds:
            i = 0
            while True:
                try:
                    location = (location_format + jsonl_location_format) % (language, language, fold, language, fold, i)
                    preprocess_corpus_file(location, language)
                    i += 1
                except FileNotFoundError:
                    break


if __name__ == '__main__':
    main()
