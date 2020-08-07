import antlr4, sys, re
# Individual languages that we want to parse
from antlr4 import ParseTreeListener, ParserRuleContext, ParseTreeWalker, TerminalNode, ErrorNode, NoViableAltException
from antlr4.error.ErrorListener import ErrorListener
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
from src.antlr4_language_parsers.python.Python3Listener import Python3Listener as pylis
from src.antlr4_language_parsers.ruby.CorundumLexer import CorundumLexer as rubyl
from src.antlr4_language_parsers.ruby.CorundumParser import CorundumParser as rubyp
from src.preprocessor.formal_lang_heuristics import is_diff_header, is_email, is_URI

from src.preprocessor.codeSearch_preprocessor import ast_to_tagged_list


class Mylistener(ErrorListener):
    def __init__(self):
        super()

    def syntaxError(self, recognizer, offendingSymbol, line, column, msg, e):
        None


class BruteParse:
    langinfo = {
        'go': {'lexer': gol, 'parser': gop, 'toprules': ['sourceFile']},
        'javascript': {'lexer': jsl, 'parser': jsp, 'toprules': ['program']},
        'php': {'lexer': phpl, 'parser': phpp, 'toprules': ['htmlDocument']},
        'python': {'lexer': pyl, 'parser': pyp, 'toprules': ['file_input', 'single_input', 'eval_input']},
        'ruby': {'lexer': rubyl, 'parser': rubyp, 'toprules': ['prog']},
        'java': {'lexer': javal, 'parser': javap, 'toprules': ['compilationUnit']}
    }

    def __init__(self):
        None

    def parse(self, lang, input):
        l = self.langinfo[lang]['lexer'](antlr4.InputStream(input), output=sys.stderr)
        stream = antlr4.CommonTokenStream(l)
        p = self.langinfo[lang]['parser'](stream, output=sys.stderr)
        p.removeErrorListeners()
        p.addErrorListener(Mylistener())
        p._errHandler = antlr4.error.ErrorStrategy.BailErrorStrategy()
        for r in self.langinfo[lang]['toprules']:
            try:
                tree = getattr(p, r)()
                return ast_to_tagged_list(tree)
            except ParseCancellationException:
                None
        return []

    def driver(self, context="blah def foo():\n\tNone bar"):
        self.parse('python', context)

class RowLabeller:

    def __init__(self):
        self.tagsForPost = {}
        self.bp = BruteParse()
        self.abstain = 0
        self.parsable = 0
        self.tokenIndex = 0
        self.total = 0

    def lookUpToken(self, language, row, tag_encoders):
        #print(self.total, self.parsable, self.abstain)
        #sys.stdout.flush()
        if row['Language'] == 'English':
            return 0
        self.total += 1
        postIdx = str(row['PostIdx'])
        context = re.escape(str(row['Context']))
        parsable = False
        if postIdx not in self.tagsForPost.keys():
            self.tagsForPost[postIdx]={}
        if context not in self.tagsForPost[postIdx].keys():
            # everytime we have a new context, we reset the token index.
            self.tokenIndex = 0
            ip = str(row['Context'])
            while len(ip.split()) > 1:
                res = self.bp.parse(language, ip)
                if res:
                    self.tagsForPost[postIdx][context] = res
                    parsable = True
                    break
                else:
                    ip = ip.partition(' ')[2]
            if not parsable:
                self.tagsForPost[postIdx][context] = []

        if self.tagsForPost[postIdx][context]:
            self.tokenIndex += 1
            if self.tokenIndex-1 < len(self.tagsForPost[postIdx][context]):
                label = self.tagsForPost[postIdx][context][self.tokenIndex - 1]
                if str(row['Token']) == label[0]:
                    sys.stdout.flush()
                    self.parsable += 1
                    return tag_encoders[language](label[1])

        #could not import ABSTAIN due to a circular dependency!
        self.abstain += 1
        return 0
