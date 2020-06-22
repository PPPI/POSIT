import gc
import gzip
import json

from tqdm import tqdm

from src.preprocessor.codeSearch_preprocessor import UNDEF

if __name__ == '__main__':
    location = './data/frequency_data/%s/frequency_data.json.gz'
    out_location = './data/frequency_data/frequency_data.json.gz'
    out_lang_location = './data/frequency_data/frequency_language_data.json.gz'

    languages = ['go', 'java', 'javascript', 'php', 'python', 'ruby']
    merged_frequency = {l: dict() for l in languages + ['English', 'uri', 'email', 'diff']}
    language_frequency = dict()

    for language in tqdm(languages):
        # Force collection of l_table between loop iterations
        l_table = None
        gc.collect()
        with gzip.open(location % language, 'rb') as f:
            # noinspection PyRedeclaration
            l_table = json.loads(f.read())

        for cp_lang in tqdm(l_table.keys(), leave=False):
            inner_table = l_table[cp_lang]
            for tok in tqdm(inner_table.keys(), leave=False):
                tok_table = inner_table[tok]

                # Language look-up
                actual_tags = len([tag for tag in tok_table.keys() if tag != UNDEF])
                if actual_tags > 0:
                    try:
                        language_frequency[tok][cp_lang] += 1
                    except KeyError:
                        language_frequency[tok] = {lang: 0 for lang in languages + ['English', 'uri', 'email', 'diff']}
                        language_frequency[tok][cp_lang] += 1

                # Tag merger
                for tag, count in tok_table.items():
                    if tag != UNDEF:
                        try:
                            merged_frequency[cp_lang][tok][tag] += count
                        except KeyError:
                            try:
                                merged_frequency[cp_lang][tok][tag] = count
                            except KeyError:
                                merged_frequency[cp_lang][tok] = dict()
                                merged_frequency[cp_lang][tok][tag] = count

    with gzip.open(out_location, 'wb') as f:
        f.write(json.dumps(merged_frequency).encode('utf-8'))

    with gzip.open(out_lang_location, 'wb') as f:
        f.write(json.dumps(language_frequency).encode('utf-8'))
