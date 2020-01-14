import json
import sys
from collections import defaultdict, Counter

from .lucid_reader import get_pattern_paths

if __name__ == '__main__':
    file_loc = sys.argv[1]
    freq_map = defaultdict(list)
    text = ''
    for lucid_file in get_pattern_paths('*.txt', file_loc):
        with open(lucid_file, encoding='utf-8') as f:
            text += '\n\n' + f.read()[:-1]
    examples = text.split('\n\n')
    examples = [e for e in examples if len(e) > 0]
    for sent in examples:
        toks = sent.split('\n')
        for tok in toks:
            idx = tok.rfind(' ')
            if idx != -1:
                word, tag = tok[:idx], tok[idx + 1:]
                freq_map[word].append(tag)
    freq_map = dict(freq_map)
    freq_normed = dict()
    for word, tags in freq_map.items():
        freq_normed[word] = dict(Counter(freq_map[word]))
    with open('./data/frequency_data.json', 'w') as f:
        f.write(json.dumps(freq_normed))
