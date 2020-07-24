import antlr4, sys
# Individual languages that we want to parse
from antlr4.error.Errors import ParseCancellationException

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
    langinfo = {
        'go': {'lexer' : gol, 'parser' : gop},
        'javascript': {'lexer' : javal, 'parser' : javap},
        'php': {'lexer' : phpl, 'parser' : phpp},
        'python': {'lexer' : pyl, 'parser' : pyp},
        'ruby': {'lexer' : rubyl, 'parser' : rubyp},
        'java': {'lexer' : javal, 'parser' : javap}
    }

    def __init__(self):
        None

    def parse(self, lang, input):
        l = self.langinfo[lang]['lexer'](antlr4.InputStream(input), output=sys.stderr)
        stream = antlr4.CommonTokenStream(l)
        p = self.langinfo[lang]['parser'](stream, output=sys.stderr)

        #TODO: p.addErrorListener(...)
        p._errHandler = antlr4.error.ErrorStrategy.BailErrorStrategy()

        for r in p.ruleNames:
            try:
                tree = getattr(p, r)()
                print(input)
                print(tree.toStringTree(None, None))
                #print(r)
                #print('Can Parse')
            except ParseCancellationException:
                continue

            #tree = p.r()
            #print(tree.toStringTree(None, None))

    def driver(self, context="blah def foo():\n\tNone bar"):
        self.parse('python', context)

'''
if __name__ == "__main__":
    p = BruteParse()
    p.driver()
'''

