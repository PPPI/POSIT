import gzip as gz
import json
import os
from collections import defaultdict, Counter

import json_lines as jl

languages = [
    'go',
    'javascript',
    'php',
    'python',
    'ruby',
    'java',
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
    'test',
    'valid',
    'train',
]
# (language, fold, language, fold, fold number)
jsonl_location_format = '%s\\final\\jsonl\\%s\\%s_%s_%d_parsed.jsonl.gz'

UNDEF = 'UNDEF'

if __name__ == '__main__':
    freq_map = {l: defaultdict(list) for l in languages + natural_languages + formal_languages}
    for language in languages:
        for fold in folds:
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
            for i in range(n_files - 1):
                location = (location_format + jsonl_location_format) % (language, language, fold, language, fold, i)
                with jl.open(location) as json_generator:
                    for json_line in json_generator:
                        for parsed in [json_line['code_parsed'], json_line['docstring_parsed']]:
                            try:
                                for tok, (source_language, tag_map) \
                                        in (eval(parsed) if isinstance(parsed, str) else parsed):
                                    for l in languages + natural_languages + formal_languages:
                                        try:
                                            freq_map[l][tok].append(tag_map[l])
                                        except KeyError:
                                            pass
                            except ValueError:
                                pass

        for l in languages + natural_languages + formal_languages:
            freq_map[l] = dict(freq_map[l])

        freq_normed = dict()
        for l in languages + natural_languages + formal_languages:
            for word, tags in freq_map[l].items():
                freq_normed[l][word] = dict(Counter(freq_map[word]))

        os.makedirs('./data/frequency_data/%s' % language, exist_ok=True)
        with gz.open('./data/frequency_data/%s/frequency_data.json.gz' % language, 'w') as f:
            f.write(json.dumps(freq_normed))
