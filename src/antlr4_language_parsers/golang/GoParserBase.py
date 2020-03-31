from antlr4 import *

from src.antlr4_language_parsers.golang.GoLexer import GoLexer

"""
All parser methods that used in grammar (p, prev, notLineTerminator, etc.)
should start with lower case char similar to parser rules.
"""
class GoParserBase(Parser):
    def __init__(self, _input, output):
        super(GoParserBase, self).__init__(_input, output)
    
    """
    Returns {@code true} iff on the current index of the parser's
    token stream a token exists on the {@code HIDDEN} channel which
    either is a line terminator, or is a multi line comment that
    contains a line terminator.
    
    @return {@code true} iff on the current index of the parser's
    token stream a token exists on the {@code HIDDEN} channel which
    either is a line terminator, or is a multi line comment that
    contains a line terminator.
    """
    def lineTerminatorAhead(self,):
        # Get the token ahead of teh current index.
        possibleIndexEosToken = self.getCurrentToken().getTokenIndex() - 1

        if possibleIndexEosToken == -1:
            return True
        
        ahead = self._input.get(possibleIndexEosToken)
        if ahead.getChannel() != Lexer.HIDDEN:
            # We're only interested in tokens on the HIDDEN channel.
            return False
        
        if ahead.getType() == GoLexer.TERMINATOR:
            # There is definitely a line terminator ahead.
            return True

        if ahead.getType() == GoLexer.WS:
            # Get the token ahead of the current whitespaces.
            possibleIndexEosToken = self.getCurrentToken().getTokenIndex() - 2
            
            ahead = self._input.get(possibleIndexEosToken)

        # Get the token's text and type
        text = ahead.getText()
        _type = ahead.getType()

        # Check if the token is, or contains a line terminator
        return (_type == GoLexer.COMMENT and (text.contains('\r') or text.contains('\n'))) or (_type == GoLexer.TERMINATOR)

    """
    Returns {@code true} if no line terminator exists between the specified
    token offset and the prior one on the {@code HIDDEN} channel.

    @return {@code true} if no line terminator exists between the specified
    token offset and the prior one on the {@code HIDDEN} channel.
    """
    def noTerminatorBetween(self, tokenOffset):
        stream = BufferedTokenStream.BufferedTokenStream(GoLexer(self._input))
        tokens = stream.getHiddenTokensToLeft(stream.LT(tokenOffset).getTokenIndex())
        
        if tokens is None:
            return True

        for token in tokens:
            if token.getText().contains("\n"):
                return False

        return True

    """
    Returns {@code true} if no line terminator exists after any encounterd
    parameters beyond the specified token offset and the next on the
    {@code HIDDEN} channel.

    @return {@code true} if no line terminator exists after any encounterd
    parameters beyond the specified token offset and the next on the
    {@code HIDDEN} channel.
    """
    def noTerminatorAfterParams(self, tokenOffset):
        stream = BufferedTokenStream.BufferedTokenStream(GoLexer(self._input))
        leftParams = 1
        rightParams = 0

        if (stream.LT(tokenOffset).getType() == GoLexer.L_PAREN):
            # Scan past parameters
            while (leftParams != rightParams):
                tokenOffset += 1
                valueType = stream.LT(tokenOffset).getType()

                if (valueType == GoLexer.L_PAREN):
                    leftParams += 1
                elif (valueType == GoLexer.R_PAREN):
                    rightParams += 1

            tokenOffset += 1
            return self.noTerminatorBetween(tokenOffset)

        return True
    
    def checkPreviousTokenText(self, text):
        stream = BufferedTokenStream.BufferedTokenStream(GoLexer(self._input))
        prevTokenText = stream.LT(1).getText()
        
        if (prevTokenText == None):
            return False
        
        return prevTokenText.equals(text)
