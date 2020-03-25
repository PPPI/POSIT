# Used to parse code to ASTs
import antlr4
# Enables reading the corpus
import json_lines as jl

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


def parse_java(entry):
    code = entry['code']
    lexer = javal(antlr4.InputStream(code))
    stream = antlr4.CommonTokenStream(lexer)
    parser = javap(stream)
    tree = parser.methodDeclaration()
    return tree.toStringTree(recog=parser)


def parse_javadoc(entry):
    docstring = entry['docstring']
    lexer = javadocl(antlr4.InputStream(docstring))
    stream = antlr4.CommonTokenStream(lexer)
    parser = javadocp(stream)
    tree = parser.documentation()
    return tree.toStringTree(recog=parser)


def parse_golang(entry):
    code = entry['code']
    lexer = gol(antlr4.InputStream(code))
    stream = antlr4.CommonTokenStream(lexer)
    parser = gop(stream)
    tree = parser.methodDecl()
    return tree.toStringTree(recog=parser)


def parse_js(entry):
    code = entry['code']
    lexer = jsl(antlr4.InputStream(code))
    stream = antlr4.CommonTokenStream(lexer)
    parser = jsp(stream)
    tree = parser.program()
    return tree.toStringTree(recog=parser)


def parse_php(entry):
    code = entry['code']
    lexer = phpl(antlr4.InputStream(code))
    stream = antlr4.CommonTokenStream(lexer)
    parser = phpp(stream)
    tree = parser.functionDeclaration()
    return tree.toStringTree(recog=parser)


def parse_python(entry):
    code = entry['code']
    lexer = pyl(antlr4.InputStream(code))
    stream = antlr4.CommonTokenStream(lexer)
    parser = pyp(stream)
    tree = parser.file_input()
    return tree.toStringTree(recog=parser)


def parse_ruby(entry):
    code = entry['code']
    lexer = rubyl(antlr4.InputStream(code))
    stream = antlr4.CommonTokenStream(lexer)
    parser = rubyp(stream)
    tree = parser.prog()
    return tree.toStringTree(recog=parser)


def read_corpus_file(location, language):
    with jl.open(location) as f:
        for entry in f:
            ast = globals()["parse_%s" % language](entry)
            if language == 'java':
                docstring_parse = parse_javadoc(entry)
            else:
                # TODO: Process the line using NLTK
                pass

            # TODO: Do something with AST and docstring to save for POSIT


def main():
    location = (location_format + jsonl_location_format) % ('ruby', 'ruby', 'train', 'ruby', 'train', 0)
    read_corpus_file(location, 'ruby')


if __name__ == '__main__':
    main()
