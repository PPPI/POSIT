import gzip
import io
import json
import re
import sys
from collections import deque
from multiprocessing.pool import Pool
from os import cpu_count

# Used to parse code to ASTs
import antlr4
# Enables reading the corpus
import json_lines as jl
# NLP tokenisation and tagging
from nltk import sent_tokenize, casual_tokenize, pos_tag
# Progress bars
from tqdm import tqdm

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
from src.preprocessor.formal_lang_heuristics import is_diff_header, is_email, is_URI

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

formal_languages = [
    'uri',
    'email',
    'diff'
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
    'valid',
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


def parse_go(entry):
    code = entry['code']
    lexer_output = io.StringIO()
    lexer = gol(antlr4.InputStream(code), output=lexer_output)
    stream = antlr4.CommonTokenStream(lexer)
    parser = gop(stream)
    tree = parser.methodDecl()
    if len(lexer_output.read()) > 0:
        tree = parser.functionDecl()
    return tree


def parse_javascript(entry):
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
    tree = parser.phpBlock()
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
            elif is_URI(tok):
                if tok.startswith('file'):
                    tag = 'file'
                elif tok.startswith('http'):
                    tag = 'http'
                elif tok.startswith('ftp'):
                    tag = 'ftp'
                elif tok.startswith('localhost'):
                    tag = 'localhost'
                elif re.match(r"\d{1,3}\.\d{1,3}\.\d{1,3}\.\d{1,3}", tag) is not None:
                    tag = 'ipv4'

                tag_built = ('uri', {l: tag_ if l == 'English' else UNDEF
                                     for l in languages + natural_languages + formal_languages})
                tag_built[-1]['uri'] = tag

                result.append(tag_built)
            elif is_email(tok):
                tag = 'email'

                tag_built = ('email', {l: tag_ if l == 'English' else UNDEF
                                     for l in languages + natural_languages + formal_languages})
                tag_built[-1]['email'] = tag

                result.append(tag_built)
            elif is_diff_header(tok):
                if tok.startswith('diff'):
                    tag = 'diff_line'
                elif tok.startswith('index'):
                    tag = 'index_line'
                else:
                    tag = 'path_line'

                tag_built = ('diff', {l: tag_ if l == 'English' else UNDEF
                                     for l in languages + natural_languages + formal_languages})
                tag_built[-1]['diff'] = tag

                result.append(tag_built)

            else:
                result.append(
                    (tok, ('English', {l: tag_ if l == 'English' else UNDEF
                                       for l in languages + natural_languages + formal_languages})))

    return result


def process_entry(entry, language):
    ast = globals()["parse_%s" % language](entry)
    tagged_code_list = [
        (tok, (language, {l: tag if l == language else UNDEF
                          for l in languages + natural_languages + formal_languages}))
        for tok, tag in ast_to_tagged_list(ast)
    ]
    tagged_docstring_list = parse_docstring(entry, language, dict(tagged_code_list))
    entry['code_parsed'] = json.dumps(tagged_code_list)
    entry['docstring_parsed'] = json.dumps(tagged_docstring_list)
    return entry


def preprocess_corpus_file(location, language):
    preprocessed_data = ''
    with jl.open(location) as f:
        for entry in tqdm(f, leave=False, desc="Entries"):
            entry = process_entry(entry, language)
            preprocessed_data += json.dumps(entry) + '\n'

    with gzip.open(location[:-len('.jsonl.gz')] + '_parsed.jsonl.gz', 'wb') as f:
        f.write(preprocessed_data.encode('utf8'))


class ProcessEntryWrapper(object):
    def __init__(self, language):
        self.language = language

    def __call__(self, entry):
        return process_entry(entry, self.language)


def main():
    for language in tqdm(languages, desc="Languages"):
        process_entry_mp = ProcessEntryWrapper(language)
        for fold in tqdm(folds, leave=False, desc="Fold"):
            # Determine number of files, we error fast, but don't actually read the file by using jl.open()
            n_files = 0
            while True:
                try:
                    location = (location_format + jsonl_location_format) \
                               % (language, language, fold, language, fold, n_files)
                    with jl.open(location) as _:
                        pass
                except FileNotFoundError:
                    break
                finally:
                    n_files += 1

            for i in tqdm(range(n_files - 1), leave=False, desc='Files'):
                # We load the full file in memory before we process it
                preprocessed_data = []
                entries = list()
                location = (location_format + jsonl_location_format) % (language, language, fold, language, fold, i)
                with jl.open(location) as f:
                    for idx, entry in enumerate(f):
                        entries.append(entry)

                with Pool(processes=cpu_count() - 1) as wp:
                    for processed_entry in tqdm(wp.imap_unordered(process_entry_mp, entries, ),
                                                leave=False, desc='Entries', total=idx + 1):
                        preprocessed_data.append(json.dumps(processed_entry))

                with gzip.open(location[:-len('.jsonl.gz')] + '_parsed.jsonl.gz', 'wb') as f:
                    f.write('\n'.join(preprocessed_data).encode('utf8'))


def main_single_threaded():
    """
    This method is mainly for debugging.
    The multithreaded main() works fine provided the underlying parsers and lexers work,
    """
    for language in tqdm(languages, desc="Languages"):
        for fold in tqdm(folds, leave=False, desc="Fold"):
            # Determine number of files, we error fast, but don't actually read the file by using jl.open()
            n_files = 0
            while True:
                try:
                    location = (location_format + jsonl_location_format) \
                               % (language, language, fold, language, fold, n_files)
                    with jl.open(location) as _:
                        pass
                except FileNotFoundError:
                    break
                finally:
                    n_files += 1

            for i in tqdm(range(n_files - 1), leave=False, desc='Files'):
                location = (location_format + jsonl_location_format) % (language, language, fold, language, fold, i)
                preprocess_corpus_file(location, language)


if __name__ == '__main__':
    if len(sys.argv) > 1:
        if sys.argv[1].lower() == 'debug':
            main_single_threaded()
    else:
        main()
