import json

import json_lines as jl
from gensim.corpora import Dictionary
from tqdm import tqdm

from src.preprocessor.codeSearch_preprocessor import location_format, folds, jsonl_location_format, languages
from src.tagger.data_utils import O

jsonl_result_location_format = jsonl_location_format[:-len('.jsonl.gz')] + '_parsed.jsonl.gz'


def generate_vocab_for_language(language):
    full_tag_list = list()
    for fold in tqdm(folds, leave=False, desc="Fold"):
        # Determine number of files, we error fast, but don't actually read the file by using jl.open()
        n_files = 0
        while True:
            try:
                location = (location_format + jsonl_result_location_format) \
                           % (language, language, fold, language, fold, n_files)
                with jl.open(location) as _:
                    pass
            except FileNotFoundError:
                break
            finally:
                n_files += 1

        for i in tqdm(range(n_files - 1), leave=False, desc='Files'):
            location = (location_format + jsonl_result_location_format) % (language, language, fold, language, fold, i)
            with jl.open(location) as f:
                for entry in tqdm(f, leave=False, desc="Entries"):
                    if isinstance(entry['code_parsed'], str):
                        code_parse = json.loads(entry['code_parsed'])
                    else:
                        code_parse = entry['code_parsed']

                    for parse in code_parse:
                        full_tag_list.append(parse[-1][-1][parse[-1][0]])

                    if isinstance(entry['docstring_parsed'], str):
                        doc_parsed = json.loads(entry['docstring_parsed'])
                    else:
                        doc_parsed = entry['docstring_parsed']

                    for parse in doc_parsed:
                        try:
                            full_tag_list.append(parse[-1][-1][parse[-1][0]])
                        except KeyError:
                            pass

    t_dct = Dictionary([full_tag_list, [O]])
    t_dct.save('./data/frequency_data/%s/tags.dct' % language)


if __name__ == '__main__':
    for language in languages:
        generate_vocab_for_language(language)
