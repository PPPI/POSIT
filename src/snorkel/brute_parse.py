import antlr4, sys
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

class BruteParse:
    lang = ['go',
            'javascript',
            'php',
            'python',
            'ruby',
            'java', ]

    def __init__(self):
        None

    def bruteParse(self, context="def foo():\n\tNone"):
        for l in self.lang:
            lexer = pyl(antlr4.InputStream(context), output=sys.stderr)
            stream = antlr4.CommonTokenStream(lexer)
            parser = pyp(stream, output=sys.stderr)
            tree = parser.funcdef()
            print(tree.toStringTree(None, None))

if __name__ == "__main__":
    p = BruteParse()
    p.bruteParse()


