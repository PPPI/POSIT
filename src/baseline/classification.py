import fileinput
import re
import sys
from multiprocessing.pool import ThreadPool

import numpy as np
from nltk import pos_tag
from sklearn import svm, metrics
from tqdm import tqdm

from ..preprocessor.preprocess import annotate_comment_line
from ..preprocessor.util import operators, line_comment_start, keywords
from ..tagger.data_utils import camel, snake, O
from ..tagger.data_utils import generate_feature_vector as feature_vector


def flip_single_english_to_code(sent):
    if len(sent) < 3:
        return sent
    for idx in range(3, len(sent)):
        sent_slice = sent[idx - 3: idx]
        if sent_slice[0] == sent_slice[-1] == 1:
            sent[idx - 2] = 1

    return sent


def heuristic_tag(tok, pos, tokens):
    if tok in keywords:
        return 'keyword'
    elif tok == '+':
        return 'plus'
    elif tok == '-':
        return 'minus'
    elif tok == ':':
        return 'colon'
    elif tok == '=':
        return 'equal'
    elif tok == '==':
        return 'equalequal'
    elif tok == '#':
        return 'hash'
    elif tok == '<':
        return 'less'
    elif tok == '<<':
        return 'lessless'
    elif tok == '>':
        return 'greater'
    elif tok == '>>':
        return 'greatergreater'
    elif tok == '->':
        return 'arrow'
    elif tok in operators or (tok[-1] == ';' and tok[:-1] in operators):
        return '.'
    elif pos + 1 < len(tokens) and '(' in tokens[pos + 1]:
        return 'raw_identifier'
    elif pos + 1 < len(tokens) and (tokens[pos + 1] in operators
                                    or (tokens[pos + 1][-1] == ';' and tokens[pos + 1][:-1] in operators)):
        return 'raw_identifier'
    elif re.match(r"\"[^\"]*\"|'[^']*'", tok):
        if len(tok) == 3:
            return 'char_constant'
        else:
            return 'string_literal'
    else:
        return O


def sentence_accuracy(list_of_sent_actual, list_of_sent_pred):
    assert len(list_of_sent_actual) == len(list_of_sent_pred)
    missed = 0
    hit = 0
    for i in range(len(list_of_sent_actual)):
        pred_ = list_of_sent_pred[i]
        actual = list_of_sent_actual[i]
        miss = False
        for p, a in zip(pred_, actual):
            if p != a:
                miss = True
                break
        if miss:
            missed += 1
        else:
            hit += 1
    return hit / (hit + missed)


def heuristic_lid(tok):
    if snake(tok):
        return 1
    elif camel(tok) and not (tok.istitle()):
        return 1
    elif any(char.isdigit() for char in tok):
        return 1
    elif set('[~!@#$%^&*()_+{}":;\']+$').intersection(tok):
        return 1
    elif any(char.isupper() for char in tok[1:]):
        return 1
    elif tok in keywords:
        return 1
    return 0


def annotate_line(tokens, context=None):
    if len(tokens) > 0 and tokens[0] in line_comment_start:
        return annotate_comment_line(tokens), context
    context = dict() if context is None else context
    result_ = list()
    for pos, tok in enumerate(tokens):
        tok = tok.strip()
        tag = heuristic_tag(tok, pos, tokens)
        if tag in ['method', 'raw_identifier']:
            context[tok] = tag
        elif tag == O:
            if tok in context.keys():
                tag = context[tok]
            elif snake(tok) or camel(tok):
                tag = 'raw_identifier'
        result_.append((tok, tag))
    return result_, context


def unpack_data(data):
    W_, X_, y_, z__ = list(), list(), list(), list()
    for sent in data:
        W_, X_, y_, z__ = list(), list(), list(), list()
        for word, fv, tag, lid in sent:
            W_.append(word)
            X_.append(fv)
            y_.append(tag)
            z__.append(lid)
        W_.append(W_)
        X_.append(X_)
        y_.append(y_)
        z__.append(z__)
    return W_, X_, y_, z__


def parse_file(file_location):
    with fileinput.input(file_location, openhook=fileinput.hook_encoded("utf-8")) as f:
        data = list()
        current = list()
        for l in f:
            if len(l.strip()) == 0:
                data.append(current)
                current = list()
            else:
                current.append(tuple(l.strip().split(' ')))

    return [[(s[0], feature_vector(s[0]), s[1], int(s[2])) for s in sent if len(s) == 3] for sent in data]


def annotate_using_lid(sent, z_pred_):
    POS_tags = pos_tag(sent, tagset="universal")
    code_tags, _ = annotate_line(sent, None)
    result_ = list()
    for i in range(len(sent)):
        language_id = z_pred_[i]
        if language_id == 1:
            result_.append(code_tags[i])
        else:
            result_.append(POS_tags[i])
    return result_


def annotate_using_rules(sent):
    POS_tags = pos_tag(sent, tagset="universal")
    code_tags, _ = annotate_line(sent, None)
    result_ = list()
    z__ = list()
    for i, word in enumerate(sent):
        z__.append(heuristic_lid(word))

    # z__ = flip_single_english_to_code(z__)

    for i, lid in enumerate(z__):
        if lid == 0:
            result_.append(POS_tags[i])
        else:
            result_.append(code_tags[i])
    return result_, z__


def wrap_callable(callable_):
    def wrapper(args):
        return callable_(*args)

    return wrapper


if __name__ == '__main__':
    dataset = sys.argv[1]
    with_svm = sys.argv[2].lower() == 'true'
    file_location_t = './data/corpora/%s/train.txt' % dataset
    file_location_d = './data/corpora/%s/dev.txt' % dataset
    file_location_e = './data/corpora/%s/eval.txt' % dataset

    W, X, y, z = unpack_data(parse_file(file_location_t))

    W_test, X_test, y_test, z_test = unpack_data(parse_file(file_location_e))
    # Perform one tag here to pre-load the tags
    pos_tag(W[0], tagset="universal")

    if with_svm:
        # SVM prediction below
        svc = svm.SVC(kernel='rbf', gamma='scale')
        svc.fit([s for sub in X for s in sub], [s for sub in z for s in sub])
        z_pred = svc.predict([s for sub in X_test for s in sub])
        svc_t = svm.SVC(kernel='rbf', gamma='scale')
        svc_t.fit([s for sub in X for s in sub], [s for sub in y for s in sub])
        y_pred_t = svc_t.predict([s for sub in X_test for s in sub])

        layered_z = list()
        offset = 0
        for _ in W_test:
            layered_z.append(z_pred[offset: offset + len(_)])
            offset += len(_)

        layered_y = list()
        offset = 0
        for _ in W_test:
            layered_y.append(y_pred_t[offset: offset + len(_)])
            offset += len(_)

        y_pred = list()
        with ThreadPool() as wp:
            for result in tqdm(wp.imap(wrap_callable(annotate_using_lid), zip(W_test, layered_z)), total=len(W_test)):
                y_pred.append(result)
        accs = list()
        for pred, real in zip(y_pred, y_test):
            accs.append(metrics.accuracy_score(real, [tok[-1] for tok in pred]))

        print('LID prediction accuracy')
        print(metrics.accuracy_score([s for sub in z_test for s in sub], z_pred))
        print('LID sentence accuracy')
        print(sentence_accuracy(z_test, layered_z))

        print('Tag prediction accuracy')
        print(np.nanmean(accs))
        print('Tag sentence accuracy')
        print(sentence_accuracy(y_test, [[tok[-1] for tok in sent] for sent in y_pred]))

        print('Direct tag prediction accuracy')
        print(metrics.accuracy_score([s for sub in y_test for s in sub], y_pred_t))
        print('Direct tag sentence accuracy')
        print(sentence_accuracy(y_test, layered_y))

    y_pred = list()
    z_pred = list()
    with ThreadPool() as wp:
        for result, z_ in tqdm(wp.imap(annotate_using_rules, W_test), total=len(W_test)):
            y_pred.append(result)
            z_pred.append(z_)
    accs = list()
    for pred, real in zip(y_pred, y_test):
        accs.append(metrics.accuracy_score(real, [tok[-1] for tok in pred]))
    print('Tag prediction accuracy w/o SVM')
    print(np.nanmean(accs))
    print('Tag sentence accuracy')
    print(sentence_accuracy(y_test, [[tok[-1] for tok in sent] for sent in y_pred]))
    print('LID prediction accuracy w/o SVM')
    print(metrics.accuracy_score([s for sub in z_test for s in sub], [s for sub in z_pred for s in sub]))
    print('LID sentence accuracy w/o SVM')
    print(sentence_accuracy(z_test, z_pred))
