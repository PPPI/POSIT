import gzip
import json

from src.preprocessor.codeSearch_preprocessor import UNDEF

if __name__ == '__main__':
    location = './data/frequency_data/%s/frequency_data.json.gz'
    out_location = './data/frequency_data/frequency_data.json.gz'
    out_lang_location = './data/frequency_data/frequency_language_data.json.gz'

    languages = ['go', 'java', 'javascript', 'php', 'python', 'ruby']
    merged_frequency = {l: dict() for l in languages + ['English', 'url', 'email', 'diff']}
    language_frequency = dict()

    for language in languages:
        with gzip.open(location % language, 'rb') as f:
            l_table = json.loads(f.read())

        for cp_lang in l_table.keys():
            inner_table = l_table[cp_lang]
            for tok in inner_table.keys():
                tok_table = inner_table[tok]

                # Language look-up
                try:
                    language_frequency[tok][cp_lang] += 1
                except KeyError:
                    language_frequency[tok] = {l: 0 for l in languages + ['English', 'url', 'email', 'diff']}
                    language_frequency[tok][cp_lang] += 1

                # Tag merger
                for tag, count in tok_table.items():
                    if tag != UNDEF:
                        try:
                            merged_frequency[cp_lang][tok][tag] += count
                        except KeyError:
                            merged_frequency[cp_lang][tok][tag] = count

    with gzip.open(out_location, 'wb') as f:
        f.write(json.dumps(merged_frequency).encode('utf-8'))

    with gzip.open(out_lang_location, 'wb') as f:
        f.write(json.dumps(language_frequency).encode('utf-8'))
