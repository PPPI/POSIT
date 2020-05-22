import gzip
import itertools
import json

import json_lines as jl
from nltk import sent_tokenize, casual_tokenize, pos_tag
from tqdm import tqdm

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

if __name__ == '__main__':
    for language in tqdm(languages):
        for fold in tqdm(folds, leave=False):
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
            for i in tqdm(range(n_files - 1), leave=False):
                location = (location_format + jsonl_location_format) % (language, language, fold, language, fold, i)
                processed_data = list()
                with jl.open(location) as json_generator:
                    for json_line in tqdm(json_generator, leave=False):
                        parsed = json_line['docstring_parsed']
                        tagged_sents = [pos_tag(casual_tokenize(s)) for s in sent_tokenize(json_line['docstring'])]
                        flat_tagged = list(itertools.chain(*tagged_sents))
                        fixed_docstring = list()
                        for i, (tok, tag_map) in enumerate(eval(parsed)):
                            if tok in ['uri', 'email', 'diff']:
                                fixed_docstring.append((flat_tagged[i][0], tag_map))
                            else:
                                fixed_docstring.append((tok, tag_map))
                        json_line['docstring_parsed'] = fixed_docstring
                        processed_data.append(json.dumps(json_line))

                with gzip.open(location, 'wb') as f:
                    f.write('\n'.join(processed_data).encode('utf8'))
