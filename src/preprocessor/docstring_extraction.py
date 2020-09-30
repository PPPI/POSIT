import os

import json_lines as jl
import numpy as np
from tqdm import tqdm

from src.preprocessor.codeSearch_preprocessor import languages, folds, location_format, jsonl_location_format


def main():
    # Fixed for reproducibility
    np.random.seed(42)

    docstrings = {l: list() for l in languages}
    for language in tqdm(languages, desc="Languages"):
        for fold in tqdm(folds, leave=False, desc="Fold"):
            # Determine number of files, we error fast, but don't actually read the file by using jl.open()
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

            for i in tqdm(range(n_files - 1), leave=False, desc='Files'):
                location = (location_format + jsonl_location_format) % (language, language, fold, language, fold, i)
                with jl.open(location) as f:
                    for entry in tqdm(f, leave=False, desc="Entries"):
                        docstrings[language].append(entry['docstring'])

    for language in tqdm(languages, desc="Languages"):
        selected_idx = list(np.random.choice(len(docstrings[language]), 30, replace=False))
        selected = list()
        for idx in selected_idx:
            selected.append(docstrings[idx])

        os.makedirs('./data/corpora/docstring/%s' % language, exist_ok=True)
        with open('./data/corpora/docstring/%s/sampled_docstring.txt', 'w') as f:
            f.write('\n<DOC-END>\n'.join(selected))


if __name__ == '__main__':
    main()
