from collections import deque

# Used to parse code to ASTs
import antlr4
# Enables reading the corpus
import json_lines as jl

# Individual languages that we want to parse
from nltk import sent_tokenize, casual_tokenize

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


def parse_docstring(entry, language, code_context):
    result = list()
    if language == 'java':
        docstring_parse = parse_javadoc(entry)
        tagged_list = ast_to_tagged_list(docstring_parse)
        # TODO: Reconstruct sentences for NLTK.
        for snippet_or_tok, tag in tagged_list:
            if tag == "":
                toks = [casual_tokenize(s) for s in sent_tokenize(snippet_or_tok)]
                for tok, tag_ in toks:
                    if tok in code_context.keys():
                        context = code_context[tok]
                        result.append((tok, ('English', context[-1]+{'English': tag_})))
            else:
                # TODO: Output the code-entity tag
                if snippet_or_tok in code_context.keys():
                    context = code_context[snippet_or_tok]
                    result.append((snippet_or_tok, ('English', context[-1]+{'English': tag})))
    else:
        toks = [casual_tokenize(s) for s in sent_tokenize(entry['docstring'])]
        for tok, tag_ in toks:
            if tok in code_context.keys():
                context = code_context[tok]
                result.append((tok, ('English', context[-1] + {'English': tag_})))

    return result


def read_corpus_file(location, language):
    with jl.open(location) as f:
        for entry in f:
            ast = globals()["parse_%s" % language](entry)
            tagged_code_list = [
                (tok, (language, {l: tag if l == language else '' for l in languages + natural_languages}))
                for tok, tag in ast_to_tagged_list(ast)]
            tagged_docstring_list = parse_docstring(entry, language, dict(tagged_code_list))


def main():
    location = (location_format + jsonl_location_format) % ('java', 'java', 'train', 'java', 'train', 0)
    read_corpus_file(location, 'java')


if __name__ == '__main__':
    main()
