import copy
import json
import os
import sys
from fnmatch import fnmatch
from statistics import mode
from typing import List

import Levenshtein as L
from nltk import sent_tokenize, casual_tokenize, pos_tag, stem

blacklist = ['l_square', 'r_square', 'l_paren', 'r_paren', 'l_brace', 'r_brace', 'period', 'question', 'semi', 'comma']
# Note that we exclude the '.' tag as we reuse it in code as well.
_UNIVERSAL_TAGS = ('VERB', 'NOUN', 'PRON', 'ADJ', 'ADV', 'ADP', 'CONJ', 'DET', 'NUM', 'PRT', 'X')
stemmer = stem.SnowballStemmer(language='english')


def get_pattern_paths(pattern: str, path: str) -> List[str]:
    """
    Find the OS paths to all files that match a pattern
    :param pattern: The regEx pattern to match the filename to
    :param path: The root folder where the pattern will be matched
    :return: A list of paths to the found files,
    empty list when no files found
    """
    files_paths = []
    for path, _, files in os.walk(path):
        for name in files:
            if fnmatch(name, pattern):
                files_paths.append(os.path.join(path, name))
    return files_paths


def fuzzy_match(key, list_, k_):
    """
    Function to fuzzy matched a stemmed word against a list of possible matches (also stemmed)
    :param key: The token we would like to fuzzy match
    :param list_: The list we would want to match against
    :param k_: The maximum number of edits allowed for a fuzzy match
    :return: The fuzzy matched list as pairs of matched token and edits
    """
    key = stemmer.stem(key)
    list_ = [(w, stemmer.stem(w)) for w in list_]
    list_ = [p for p in sorted([(w, L.distance(key, l)) for w, l in list_], key=lambda p: p[-1]) if p[-1] <= k_]
    return list_


def parse_cc(lucid_data_, cc_block, file_out, with_l_id_, freq_context_=None, fuzzy_k=3):
    observed_ = set()
    cc_block_children = [k_ for k_ in cc_block.keys() if k_.startswith('Child')]
    for cc_block_child_key in cc_block_children:
        cc_block_child = cc_block[cc_block_child_key]
        if cc_block_child.startswith('Snippet'):
            for annotated_line in cc_block[cc_block_child]:
                line = [tuple([p[0], p[1] if p[1] not in blacklist else '.'] + ([1] if with_l_id_ else []))
                        for p in annotated_line['Tokens']]
                line_2 = copy.deepcopy(line)
                code_toks = [(p[0], p[1]) for p in copy.deepcopy(line) if p[1] != "comment" and p[1] != "keyword"]
                if with_l_id_:
                    for val, tag, l_id in line:
                        if tag == "comment":
                            toks = [
                                pos_tag([t for t in casual_tokenize(s) if t not in ['/', '\\', '*']],
                                        tagset="universal")
                                for s in sent_tokenize(cc_block['CommentText'].strip())]
                            new_toks = list()
                            for sent in toks:
                                new_sent = list()
                                for val_, tag_ in sent:
                                    try:
                                        short_list = fuzzy_match(val_, [p[0] for p in code_toks], fuzzy_k)
                                        just_list = [p[0] for p in short_list]
                                        new_tag = mode([p[1] for p in code_toks if p[0] in just_list])
                                        if new_tag.isupper():
                                            new_tag = tag_
                                        new_sent.append(
                                            (val_, new_tag, 0 if new_tag.isupper() or new_tag == '.' else 1))
                                    except IndexError:
                                        new_sent.append((val_, tag_, 0))
                                new_toks.append(new_sent)
                            toks = new_toks
                            if freq_context_ is not None:
                                new_toks = list()
                                for sent in toks:
                                    new_sent = list()
                                    for val_, tag_ in sent:
                                        try:
                                            new_tag = \
                                                sorted(freq_context_[val_].items(), reverse=True,
                                                       key=lambda p: p[1])[0][0]
                                            if new_tag.isupper():
                                                new_tag = tag_
                                            new_sent.append(
                                                (val_, new_tag, 0 if new_tag.isupper() or new_tag == '.' else 1))
                                        except IndexError:
                                            new_sent.append((val_, tag_, 0))
                                    new_toks.append(new_sent)
                                toks = new_toks
                            formatted_output = ''.join(['\n'.join(['%s %s %d' % t for t in s]) + '\n\n' for s in toks])
                            file_out.write(formatted_output)
                            line_2.remove((val, tag, l_id))
                        elif tag == "string_literal":
                            line_2.remove((val, tag, l_id))
                            line_2.append((val.replace('\n', ' '), tag, l_id))
                    file_out.write('\n'.join(['%s %s %d' % (v, t, l) for v, t, l in line_2]) + '\n\n')
                else:
                    for val, tag in line:
                        if tag == "comment":
                            toks = [
                                pos_tag([t for t in casual_tokenize(s) if t not in ['/', '\\', '*']],
                                        tagset="universal")
                                for s in sent_tokenize(cc_block['CommentText'].strip())]
                            new_toks = list()
                            for sent in toks:
                                new_sent = list()
                                for val_, tag_ in sent:
                                    try:
                                        short_list = fuzzy_match(val_, [p[0] for p in code_toks], fuzzy_k)
                                        just_list = [p[0] for p in short_list]
                                        new_tag = mode([p[1] for p in code_toks if p[0] in just_list])
                                        if new_tag.isupper():
                                            new_tag = tag_
                                        new_sent.append(
                                            (val_, new_tag, 0 if new_tag.isupper() or new_tag == '.' else 1))
                                    except IndexError:
                                        new_sent.append((val_, tag_, 0))
                                new_toks.append(new_sent)
                            toks = new_toks
                            if freq_context_ is not None:
                                new_toks = list()
                                for sent in toks:
                                    new_sent = list()
                                    for val_, tag_ in sent:
                                        try:
                                            new_tag = \
                                                sorted(freq_context_[val_].items(), reverse=True, key=lambda p: p[1])[
                                                    0][
                                                    0]
                                            if new_tag.isupper():
                                                new_tag = tag_
                                            new_sent.append((val_, new_tag))
                                        except IndexError:
                                            new_sent.append((val_, tag_))
                                    new_toks.append(new_sent)
                                toks = new_toks
                            formatted_output = ''.join(['\n'.join(['%s %s' % t for t in s]) + '\n\n' for s in toks])
                            file_out.write(formatted_output)
                            line_2.remove((val, tag))
                        elif tag == "string_literal":
                            line_2.remove((val, tag))
                            line_2.append((val.replace('\n', ' '), tag))
                    file_out.write('\n'.join(['%s %s' % (v, t) for v, t in line_2]) + '\n\n')
        else:
            observed_.add(cc_block_child)
            inner_observed = parse_cc(lucid_data_, lucid_data_[cc_block_child], file_out, with_l_id_)
            observed_.union(inner_observed)
    if len(cc_block['CommentText']) > 0:
        toks = [pos_tag([t for t in casual_tokenize(s) if t not in ['/', '\\', '*']], tagset="universal")
                for s in sent_tokenize(cc_block['CommentText'].strip())]
        if freq_context_ is not None:
            if with_l_id_:
                new_toks = list()
                for sent in toks:
                    new_sent = list()
                    for val, tag in sent:
                        try:
                            new_tag = sorted(freq_context_[val].items(), reverse=True, key=lambda p: p[1])[0][0]
                            if new_tag.isupper():
                                new_tag = tag
                            new_sent.append((val, new_tag, 0 if new_tag.isupper() or new_tag == '.' else 1))
                        except IndexError:
                            new_sent.append((val, tag, 0))
                    new_toks.append(new_sent)
                toks = new_toks
            else:
                new_toks = list()
                for sent in toks:
                    new_sent = list()
                    for val, tag in sent:
                        try:
                            new_tag = sorted(freq_context_[val].items(), reverse=True, key=lambda p: p[1])[0][0]
                            if new_tag.isupper():
                                new_tag = tag
                            new_sent.append((val, new_tag))
                        except IndexError:
                            new_sent.append((val, tag))
                    new_toks.append(new_sent)
                toks = new_toks
        if with_l_id_:
            formatted_output = ''.join(['\n'.join(['%s %s %d' % t for t in s]) + '\n\n' for s in toks])
        else:
            formatted_output = ''.join(['\n'.join(['%s %s' % t for t in s]) + '\n\n' for s in toks])
        file_out.write(formatted_output)
    return observed_


if __name__ == '__main__':
    file_loc = sys.argv[1]
    with_l_id = sys.argv[2].lower() == 'true'
    with open('./data/corpora/SO/frequency_map.json', encoding='utf-8') as f:
        freq_context = json.loads(f.read())
    for k in range(20):
        for lucid_file in get_pattern_paths('*.lucid', file_loc):
            # noinspection PyBroadException
            try:
                with open(lucid_file, encoding='utf-8') as f:
                    lucid_data = json.loads(f.read())

                code_comment_keys = [k for k in lucid_data.keys() if k.startswith('CodeComment')]
                visited = set()
                with open(lucid_file[:-len('.lucid')]
                          + ('_l_id' if with_l_id else '') + '%d.txt' % k, 'w', encoding='utf-8') as fo:
                    for cc_key in code_comment_keys:
                        if cc_key in visited:
                            continue
                        visited.add(cc_key)
                        observed = parse_cc(lucid_data, lucid_data[cc_key], fo, with_l_id, None, fuzzy_k=k)
                        visited.union(observed)
            except Exception:
                pass
