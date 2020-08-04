import antlr4, sys
# Individual languages that we want to parse
from antlr4 import ParseTreeListener, ParserRuleContext, ParseTreeWalker, TerminalNode
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


class Mylistener(ParseTreeListener):
    def __init__(self):
        super()

    # def exitEveryRule(self, ctx:ParserRuleContext):
    #    print("rule entered: " + ctx.getText(), type(ctx));

    def visitTerminal(self, node: TerminalNode):
        print(node)


class MyPythonListener(pylis):
    def __init__(self):
        super()

    def visitTerminal(self, node: TerminalNode):
        print(node)


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

        # TODO: p.addErrorListener(...)
        p._errHandler = antlr4.error.ErrorStrategy.BailErrorStrategy()
        walker = ParseTreeWalker()
        parsable = False
        for r in self.langinfo[lang]['toprules']:
            try:
                tree = getattr(p, r)()
                parsable = True
                # print(ast_to_tagged_list(tree))
                # if lang == 'python':
                #    listener = MyPythonListener()
                #    walker.walk(listener, tree)
                return ast_to_tagged_list(tree)
            except ParseCancellationException:
                pass
        return ['ABSTAIN']

        '''
        for r in p.ruleNames:
            try:
                tree = getattr(p, r)()
                walker = ParseTreeWalker()
                #print(input)
                #print(tree.toStringTree(None, None))
                walker.walk(listener, tree)
                print(r)
                #print('Can Parse')
            except ParseCancellationException:
                continue

            #tree = p.r()
            #print(tree.toStringTree(None, None))
        '''

    def driver(self, context="blah def foo():\n\tNone bar"):
        self.parse('python', context)


'''
if __name__ == "__main__":
    p = BruteParse()
    p.driver()
'''


class RowLabeller:

    def __init__(self):
        self.tagsForPost = {}

    def lookUpToken(self, language, row):
        if row['PostIdx'] not in self.tagsForPost.keys():
            bp = BruteParse()
            self.tagsForPost[row['PostIdx']] = bp.parse(language, row['Context'])
            #for k, v in self.tagsForPost.items():
            #    print(k, v)
            #sys.stdout.flush()
        return 0
