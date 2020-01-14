import json
import os
import re
import sys

import sklearn
from nltk import flatten, pos_tag

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


def process_natural_text(body):
    toks_ = [
        [l.strip()
         for l in re.findall(CODE_TOKENISATION_REGEX,
                             line.strip())
         if len(l.strip()) > 0]
        for line in body.split('\n')
    ]
    toks_ = flatten(toks_)
    return [(w, t, 0) for w, t in pos_tag(toks_, tagset='universal')]


def serialise_result(list_of_nodes):
    if not (isinstance(list_of_nodes, list)):
        list_of_nodes = [list_of_nodes]
    result = list()  # [(tok, tag, id)]
    for node in list_of_nodes:
        if node['type'] == 'TextFragmentNode':
            result += process_natural_text(node['text'])
        elif node['type'] == 'EmbeddedContentsNode':
            pass  # trait
        elif node['type'] == 'GenericEmbeddedContentsNode':
            # Generic container
            result += serialise_result(node['nodes'])
        elif node['type'] == 'HASTNodeSequence':
            result += serialise_result(node['comments'])
            result += serialise_result(node['fragments'])
        elif node['type'] == 'ASTTerminalNode':
            pass  # trait
        elif node['type'] == 'AbsentBodyNode':
            result += (node['comma']['symbol'], '.', 1)
        elif node['type'] == 'AnnotationElement':
            pass  # trait
        elif node['type'] == 'AnnotationMethodNode':
            result += [(keyword['value'], 'keyword', 1) for keyword in node['keywordModifiers']]
            if node['identifier']['isLikelyFieldOrMethodIdentifier']:
                tag_ = 'method_name'
            elif node['identifier']['isLikelyConstantIdentifier']:
                tag_ = 'const'
            else:
                tag_ = 'raw_identifier'
            result += [(node['identifier']['name'], tag_, 1)]
        elif node['type'] == 'AnnotationMethodOrConstantNode':
            result += [(keyword['value'], 'keyword', 1) for keyword in node['keywordModifiers']]
            if node['identifier']['isLikelyFieldOrMethodIdentifier']:
                tag_ = 'method_name'
            elif node['identifier']['isLikelyConstantIdentifier']:
                tag_ = 'const'
            else:
                tag_ = 'raw_identifier'
            result += [(node['identifier']['name'], tag_, 1)]
        elif node['type'] == 'AnnotationNode':
            result += serialise_result(node['comments'])
            if 'element' in node.keys():
                result += serialise_result([node['element']])
            result += serialise_result([node['identifier']])
        elif node['type'] == 'AnnotationTypeDeclarationNode':
            pass  # Skipped for now, not in our results
        elif node['type'] == 'ArrayAccessSelectorNode':
            result += serialise_result([node['baseExpression']])
            result += serialise_result([node['dimExpression']])
        elif node['type'] == 'ArrayCreatorNode':
            result += serialise_result([node['creatorType']])
            result += serialise_result(node['arrayDepth'])
            result += serialise_result([node['initializer']])
        elif node['type'] == 'ArrayDimensionNode':
            result += [('[', '.', 1)]
            result += [(']', '.', 1)]
        elif node['type'] == 'ArrayExpressionCreatorNode':
            result += serialise_result([node['creatorType']])
            result += serialise_result(node['expressionDepth'])
            result += serialise_result(node['arrayDepth'])
            if 'initializer' in node.keys():
                result += serialise_result([node['initializer']])
        elif node['type'] == 'ArrayInitializerNode':
            result += serialise_result(node['inits'])
        elif node['type'] == 'ArrayTypeNode':
            result += serialise_result([node['typeNode']])
            result += serialise_result(node['depth'])
        elif node['type'] == 'AssertStatementNode':
            result += serialise_result([node['mainExpression']])
        elif node['type'] == 'BaseReferenceTypeNode':
            result += serialise_result(node['ids'])
        elif node['type'] == 'BinaryExpressionNode':
            result += serialise_result([node['left']])
            result += serialise_result([node['operator']])
            result += serialise_result([node['right']])
        elif node['type'] == 'BlockDeclarationNode':
            result += serialise_result([node['block']])
        elif node['type'] == 'BlockNode':
            result += serialise_result(node['leftComments'])
            result += serialise_result(node['statements'])
            result += serialise_result(node['rightComments'])
        elif node['type'] == 'BlockStatementNode':
            pass  # Not in our results, skipped
        elif node['type'] == 'BooleanLiteralNode':
            result += [(node['valueRep'], 'boolean_const', 1)]
        elif node['type'] == 'BoundNode':
            pass  # trait
        elif node['type'] == 'BreakStatementNode':
            result += [('break', 'keyword', 1), (';', '.', 1)]
        elif node['type'] == 'CastExpressionNode':
            result += [('(', '.', 1)]
            result += serialise_result(node['types'])
            result += [(')', '.', 1)]
            result += serialise_result([node['argument']])
        elif node['type'] == 'CatchClauseNode':
            result += [('catch', 'keyword', 1), ('(', '.', 1)]
            result += serialise_result([node['catchType']])
            if 'identifiers' in node.keys():
                result += serialise_result(node['identifiers'])
            result += [(')', '.', 1)]
            result += serialise_result([node['block']])
        elif node['type'] == 'CatchTypeNode':
            result += serialise_result(node['identifiers'])
        elif node['type'] == 'CharacterLiteralNode':
            result += [(node['valueRep'], 'char_const', 1)]
        elif node['type'] == 'ClassBodyNode':
            result += serialise_result(node['leftComments'])
            result += serialise_result(node['declarations'])
            result += serialise_result(node['rightComments'])
        elif node['type'] == 'ClassDeclarationNode':
            result += serialise_result(node['modifiers'])
            if 'identifiers' in node.keys():
                result += serialise_result(node['identifiers'])
        elif node['type'] == 'ClassLiteralExpressionNode':
            pass  # Not in result set, skipped
        elif node['type'] == 'ClassRelationshipNode':
            result += serialise_result([node['typeName']])
            result += [('<', '.', 1)]
            result += serialise_result([node['superTypeName']])
        elif node['type'] == 'CommentNode':
            result += process_natural_text(node['text'])
        elif node['type'] == 'InlineCommentNode':
            if 'embeddedContents' in node.keys():
                result += serialise_result([node['embeddedContents']])
            else:
                result += process_natural_text(node['rawText'])
        elif node['type'] == 'CompilationUnitNode':
            if 'packageDeclaration' in node.keys():
                result += serialise_result(node['packageDeclaration'])
            result += serialise_result(node['imports'])
            result += serialise_result(node['typeDeclarations'])
        elif node['type'] == 'CompilationUnitPreambleNode':
            pass  # not in result set, skipped
        elif node['type'] == 'ConditionalExpressionNode':
            result += [('if', 'keyword', 1)]
            result += serialise_result([node['condition']])
            result += [('then', 'keyword', 1)]
            result += serialise_result([node['thenExp']])
            if 'elseExp' in node.keys():
                result += [('else', 'keyword', 1)]
                result += serialise_result([node['elseExp']])
        elif node['type'] == 'ConstructorCreatorNode':
            result += serialise_result([node['creatorType']])
            result += serialise_result(node['arguments'])
        elif node['type'] == 'ContinueStatementNode':
            result += [('continue', 'keyword', 1), (';', '.', 1)]
        elif node['type'] == 'CreatorNode':
            pass  # trait
        elif node['type'] == 'DimExpressionNode':
            result += [('[', '.', 1)]
            result += serialise_result([node['argument']])
            result += [(']', '.', 1)]
        elif node['type'] == 'DoWhileStatementNode':
            result += serialise_result([node['statement']])
            result += serialise_result([node['expression']])
        elif node['type'] == 'ElementValueArrayInitializerNode':
            result += serialise_result([node['values']])
        elif node['type'] == 'ElementValueNode':
            result += serialise_result([node['value']])
        elif node['type'] == 'ElementValuePairNode':
            result += serialise_result([node['identifier']])
            result += serialise_result([node['value']])
        elif node['type'] == 'ElementValuePairsNode':
            result += serialise_result(node['valuePairs'])
        elif node['type'] == 'ElementValuesNode':
            result += serialise_result(node['values'])
        elif node['type'] == 'ElseStatementNode':
            result += serialise_result([node['statement']])
        elif node['type'] == 'EmptyDeclarationNode':
            pass  # not in result set, skipped
        elif node['type'] == 'EmptyStatementNode':
            result += serialise_result([node['comma']])
        elif node['type'] == 'EnumBodyNode':
            result += serialise_result(node['members'])
        elif node['type'] == 'EnumConstantNode':
            result += serialise_result(node['modifiers'])
            result += serialise_result([node['identifier']])
        elif node['type'] == 'EnumDeclarationNode':
            result += serialise_result(node['modifiers'])
            result += serialise_result([node['identifier']])
            result += serialise_result([node['body']])
        elif node['type'] == 'ExpressionListNode':
            result += serialise_result(node['arguments'])
        elif node['type'] == 'ExpressionNode':
            pass  # trait
        elif node['type'] == 'ExpressionStatement':
            result += serialise_result([node['expression']])
        elif node['type'] == 'ExtendsTypeBoundNode':
            result += serialise_result(node['types'])
        elif node['type'] == 'FieldDeclarationNode':
            result += serialise_result(node['modifiers'])
            result += serialise_result(node['variables'])
        elif node['type'] == 'FloatingPointLiteralNode':
            result += [(node['valueRep'], 'numeric_const', 1)]
        elif node['type'] == 'ForControlNode':
            pass  # not in result set, skipped
        elif node['type'] == 'ForEachControlNode':
            result += [('(', '.', 1)]
            result += serialise_result([node['variable']])
            result += [(':', '.', 1)]
            result += serialise_result([node['expression']])
            result += [(')', '.', 1)]
        elif node['type'] == 'ForExpressionControlNode':
            result += [('(', '.', 1)]
            result += serialise_result(node['init'])
            result += [(';', '.', 1)]
            if 'condition' in node.keys():
                result += serialise_result([node['condition']])
            result += [(';', '.', 1)]
            result += serialise_result(node['update'])
            result += [(')', '.', 1)]
        elif node['type'] == 'ForLoopStatementNode':
            result += [('for', 'keyword', 1)]
            result += serialise_result([node['forControl']])
            result += serialise_result([node['statement']])
        elif node['type'] == 'ForVarControlNode':
            result += [('(', '.', 1)]
            result += serialise_result([node['variables']])
            result += [(';', '.', 1)]
            result += serialise_result([node['condition']])
            result += [(';', '.', 1)]
            result += serialise_result(node['update'])
            result += [(')', '.', 1)]
        elif node['type'] == 'FormalParameterDeclNode':
            result += serialise_result(node['modifiers'])
            result += serialise_result([node['parameterType']])
            result += serialise_result([node['variableDeclaration']])
        elif node['type'] == 'FormalParametersNode':
            result += serialise_result(node['parameters'])
        elif node['type'] == 'IdentifierNode':
            result += serialise_result(node['comments'])
            result += [(node['name'], 'raw_identifier', 1)]
        elif node['type'] == 'IdentifierWithNonWildCardTypeArgumentNode':
            pass  # not in the result set, skipped
        elif node['type'] == 'IfStatementNode':
            result += [('if', 'keyword', 1), ('(', '.', 1)]
            result += serialise_result([node['expression']])
            result += [(')', '.', 1)]
            result += serialise_result([node['statement']])
        elif node['type'] == 'ImportDeclarationNode':
            result += serialise_result(node['leftComments'])
            if node['isStatic']:
                result += [('static', 'keyword', 1)]
            result += serialise_result([node['identifier']])
            result += serialise_result(node['rightComments'])
        elif node['type'] == 'InnerConstructorInvocationSelectorNode':
            pass  # not in result set, skipped
        elif node['type'] == 'IntegerLiteralNode':
            result += [(node['valueRep'], 'numeric_const', 1)]
        elif node['type'] == 'InterfaceDeclarationNode':
            result += serialise_result(node['modifiers'])
            result += serialise_result([node['identifier']])
        elif node['type'] == 'InterfaceRelationshipNode':
            result += serialise_result([node['typeName']])
            result += [('<', '.', 1)]
            result += serialise_result([node['superTypeName']])
        elif node['type'] == 'JavaASTNode':
            pass  # not in result set, skipped
        elif node['type'] == 'KeywordModifierNode':
            result += [(node['value'], 'keyword', 1)]
        elif node['type'] == 'KeywordNode':
            pass  # trait
        elif node['type'] == 'LabelNode':
            result += serialise_result(node['comments'])
            result += serialise_result([node['identifier']])
        elif node['type'] == 'LambdaBodyBlockNode':
            pass  # not in result set, skipped
        elif node['type'] == 'LambdaBodyExpressionNode':
            result += serialise_result([node['expressionBody']])
        elif node['type'] == 'LambdaBodyNode':
            pass  # not in result set, skipped
        elif node['type'] == 'LambdaExpressionNode':
            result += serialise_result([node['lambdaParameters']])
            result += serialise_result([node['lambdaBody']])
        elif node['type'] == 'LambdaParametersNode':
            if node['hasParentheses']:
                result += [('(', '.', 1)]
            result += serialise_result(node['parameterList'])
            if node['hasParentheses']:
                result += [(')', '.', 1)]
        elif node['type'] == 'LiteralNode':
            result += serialise_result([node['node']])
        elif node['type'] == 'LiteralRepresentationNode':
            pass  # not in result set, skipped
        elif node['type'] == 'LocalVariableDeclarationStatementNode':
            result += serialise_result(node['modifiers'])
            result += serialise_result(node['declarations'])
        elif node['type'] == 'MemberAccessNode':
            pass  # not in result set, skipped
        elif node['type'] == 'MemberDeclarationNode':
            pass  # not in result set, skipped
        elif node['type'] == 'MethodBodyNode':
            pass  # not in result set, skipped
        elif node['type'] == 'MethodDeclarationNode':
            result += serialise_result(node['modifiers'])
            if 'returnType' in node.keys():
                if len(node['returnType'].keys()) == 0:
                    result += [('void', 'keyword', 1)]
                else:
                    result += serialise_result([node['returnType']])
            result += serialise_result([node['identifier']])
            result += serialise_result([node['parameters']])
        elif node['type'] == 'MethodInvocationNode':
            result += serialise_result([node['identifier']])
            result += serialise_result(node['arguments'])
        elif node['type'] == 'MethodReferenceNode':
            result += serialise_result([node['prefix']])
            result += serialise_result([node['identifier']])
        elif node['type'] == 'MissingExpressionNode':
            pass  # not in result set, skipping
        elif node['type'] == 'ModifierNode':
            pass  # not in result set, skipping
        elif node['type'] == 'NullLiteralNode':
            result += [('null', 'keyword', 1)]
        elif node['type'] == 'OperatorNode':
            result += [(node['symbol'], 'op', 1)]
        elif node['type'] == 'PackageDeclarationNode':
            result += serialise_result(node['leftComments'])
            result += serialise_result(node['annotations'])
            result += serialise_result([node['identifier']])
            result += serialise_result(node['rightComments'])
        elif node['type'] == 'ParameterizedTypeNode':
            result += serialise_result([node['identifier']])
        elif node['type'] == 'PostfixOperatorExpressionNode':
            result += serialise_result([node['argument']])
            result += serialise_result([node['operator']])
        elif node['type'] == 'PrefixOperatorExpressionNode':
            result += serialise_result([node['operator']])
            result += serialise_result([node['argument']])
        elif node['type'] == 'PrimitiveTypeNode':
            type_ = node['primitiveType']['type'].split('$')[-2].lower()
            result += [(type_, 'keyword', 1)]
        elif node['type'] == 'QualifiedIdentifierNode':
            result += serialise_result(node['identifiers'])
        elif node['type'] == 'ReferenceTypeBoxedExpression':
            result += serialise_result([node['referenceType']])
        elif node['type'] == 'ReferenceTypeNode':
            pass  # trait
        elif node['type'] == 'ResourceNode':
            result += serialise_result(node['modifier'])
            result += serialise_result([node['declaration']])
            result += serialise_result([node['expression']])
        elif node['type'] == 'ResourceSpecificationNode':
            result += serialise_result(node['resources'])
        elif node['type'] == 'ReturnStatementNode':
            result += serialise_result(node['comments'])
            result += serialise_result(node['labels'])
            if 'expression' in node.keys():
                result += serialise_result([node['expression']])
            else:
                result += [('return', 'keyword', 1), (';', '.', 1)]
        elif node['type'] == 'SeparatorNode':
            result += [(node['symbol'], '.', 1)]
        elif node['type'] == 'StringLiteralNode':
            result += [(node['valueRep'], 'string_literal', 1)]
        elif node['type'] == 'SuperTypeBoundNode':
            pass  # not in result set, skipped
        elif node['type'] == 'SwitchBlockStatementGroupNode':
            result += serialise_result(node['labels'])
            result += serialise_result(node['statements'])
        elif node['type'] == 'SwitchDefaultLabel':
            pass  # not in result set, skipped
        elif node['type'] == 'SwitchEnumConstantLabel':
            pass  # not in result set, skipped
        elif node['type'] == 'SwitchExpressionLabel':
            result += serialise_result([node['expression']])
        elif node['type'] == 'SwitchLabelNode':
            pass  # not in result set, skipped
        elif node['type'] == 'SwitchStatementNode':
            result += serialise_result([node['expression']])
            result += serialise_result(node['statements'])
        elif node['type'] == 'SynchronizedStatementNode':
            pass  # not in result set, skipped
        elif node['type'] == 'ThrowStatementNode':
            result += serialise_result([node['expression']])
        elif node['type'] == 'ThrownExceptionsNode':
            result += serialise_result(node['exceptionIdentifiers'])
        elif node['type'] == 'TryCatchStatementNode':
            result += [('try', 'keyword', 1)]
            result += serialise_result([node['block']])
            result += serialise_result(node['catches'])
        elif node['type'] == 'TypeArgumentsNode':
            result += serialise_result(node['types'])
        elif node['type'] == 'TypeDeclarationNode':
            pass  # not in result set, skipped
        elif node['type'] == 'TypeListNode':
            result += serialise_result(node['types'])
        elif node['type'] == 'TypeNode':
            pass  # trait
        elif node['type'] == 'TypeParameterNode':
            result += serialise_result([node['identifier']])
            if 'bound' in node.keys():
                result += serialise_result([node['bound']])
        elif node['type'] == 'TypeParametersNode':
            result += serialise_result(node['typeParameters'])
        elif node['type'] == 'UnboundConditionalExpressionNode':
            pass  # not in result set, skipped
        elif node['type'] == 'UndefinedTypeNode':
            pass  # not in result set, skipped
        elif node['type'] == 'VariableAssignmentNode':
            pass  # not in result set, skipped
        elif node['type'] == 'VariableDeclarationNode':
            if len(node['varType']) == 0:
                result += [('void', 'keyword', 1)]
            else:
                result += serialise_result([node['varType']])
            if 'identifier' in node.keys():
                result += serialise_result([node['identifier']])
        elif node['type'] == 'VariableExpressionInitializerNode':
            result += serialise_result(node['expression'])
        elif node['type'] == 'WhileStatementNode':
            result += [('while', 'keyword', 1), ('(', '.', 1)]
            result += serialise_result([node['expression']])
            result += [(')', '.', 1)]
            if 'body' in node.keys():
                result += serialise_result([node['body']])
        elif node['type'] == 'WildcardTypeArgumentNode':
            pass  # not in result set, skipped
        elif node['type'] == 'JsonASTNode':
            pass
        elif node['type'] == 'JsonArrayNode':
            pass
        elif node['type'] == 'JsonBooleanNode':
            pass
        elif node['type'] == 'JsonFloatNode':
            result += [(node['value'], 'numeric_constant', 1)]
        elif node['type'] == 'JsonIntNode':
            result += [(node['value'], 'numeric_constant', 1)]
        elif node['type'] == 'JsonLiteralNode':
            pass
        elif node['type'] == 'JsonMemberNode':
            result += serialise_result([node['name']])
            result += serialise_result([node['value']])
        elif node['type'] == 'JsonNullNode':
            pass
        elif node['type'] == 'JsonObjectNode':
            result += serialise_result(node['members'])
        elif node['type'] == 'JsonStringNode':
            result += [(node['value'], 'string_literal', 1)]
        elif node['type'] == 'JsonValueNode':
            pass
        elif node['type'] == 'ExceptionMessageNode':
            pass
        elif node['type'] == 'SourceLocationNode':
            pass
        elif node['type'] == 'StackTraceASTNode':
            pass
        elif node['type'] == 'StackTraceLineNode':
            pass
        elif node['type'] == 'StackTraceMoreExceptions':
            pass
        elif node['type'] == 'StackTraceNode':
            if 'threadName' in node.keys():
                result += serialise_result([node['threadName']])
            result += serialise_result([node['exceptionName']])
            if 'message' in node.keys():
                result += process_natural_text(node['message'])
        elif node['type'] == 'StackTraceThreadNameNode':
            result += [(node['threadName'], 'string_literal', 1)]
        elif node['type'] == 'XmlASTNode':
            pass
        elif node['type'] == 'XmlAttributeNode':
            pass
        elif node['type'] == 'XmlCDATANode':
            if isinstance(node['contents'], str):
                result += [(node['contents'], 'string_literal', 1)]
            else:
                result += serialise_result([node['contents']])
        elif node['type'] == 'XmlCommentNode':
            result += process_natural_text(node['contents'])
        elif node['type'] == 'XmlComposedNode':
            result += serialise_result([node['name']])
            result += serialise_result(node['attributes'])
            result += serialise_result(node['elements'])
        elif node['type'] == 'XmlDocumentNode':
            pass
        elif node['type'] == 'XmlElementNode':
            pass
        elif node['type'] == 'XmlIslandNode':
            pass
        elif node['type'] == 'XmlNameNode':
            result += [(node['value'], 'string_literal', 1)]
        elif node['type'] == 'XmlPrologNode':
            pass
        elif node['type'] == 'XmlSingleNode':
            result += serialise_result([node['name']])
            result += serialise_result(node['attributes'])
        elif node['type'] == 'XmlTextNode':
            if isinstance(node['contents'], str):
                result += [(node['contents'], 'string_literal', 1)]
            else:
                result += serialise_result([node['contents']])
    return result


if __name__ == '__main__':
    dataset = sys.argv[1]  # Location of StORMeD query results

    toks_all = list()
    ids_all = list()
    tags_all = list()

    ids_true = list()
    tags_true = list()
    toks_true = list()

    for pos in range(1000):
        try:
            with open(os.path.join(dataset, 'stormed_%d.json' % pos)) as f:
                query_result = json.loads(f.read())  # This is a HAST, we need to parse it to a vector of IDs and TAGs
            with open(os.path.join(dataset, 'stormed_%d_expected.json' % pos)) as f:
                ids_true += json.loads(f.read())  # This is a vector of IDs
            with open(os.path.join(dataset, 'stormed_%d_expected_tags.json' % pos)) as f:
                tags_true += json.loads(f.read())  # This is a vector of TAGs
            with open(os.path.join(dataset, 'stormed_%d_toks.json' % pos)) as f:
                toks_true += json.loads(f.read())  # This is a vector of TAGs

            toks, tags, ids = zip(*serialise_result(query_result))
            toks_all += toks
            ids_all += ids
            tags_all += tags

        except FileNotFoundError:
            if pos <= 998:  # In case chunk-ing resulted in 999 files
                print('You have missing files, couldn\'t find id == %d' % pos, file=sys.stderr)

    offset = 0
    comparable_tags = list()
    comparable_tags_s = list()
    comparable_ids = list()
    comparable_ids_s = list()
    for pos, (tok, tag, id_) in enumerate(zip(toks_all, tags_all, ids_all)):
        if pos + offset >= len(toks_true):
            break
        real_tok = toks_true[pos + offset]
        if tag == 'string_literal' and len(tok) > 10 and real_tok != tok:
            # offset += len(tok.split())
            continue
        look_ahead = 0
        skip = False
        while (real_tok != tok) and not (real_tok == '""' and tok == '" "'):
            offset += 1
            real_tok = toks_true[min(len(toks_true) - 1, pos + offset)]
            look_ahead += 1
            if look_ahead > 10000:
                offset -= look_ahead
                skip = True
                break
        if skip:
            continue
        comparable_tags.append(tags_true[pos + offset])
        comparable_tags_s.append(tag)
        comparable_ids.append(ids_true[pos + offset])
        comparable_ids_s.append(id_)

    tag_bacc = sklearn.metrics.balanced_accuracy_score(comparable_tags_s, comparable_tags)
    lid_bacc = sklearn.metrics.balanced_accuracy_score(comparable_ids_s, comparable_ids)

    print('Balanced Accuracy for LID: %2.3f; Tags: %2.3f.' % (lid_bacc, tag_bacc))
