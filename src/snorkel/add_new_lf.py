import os
import sys

import h5py
import numpy as np
from snorkel.labeling import PandasLFApplier
from tqdm import tqdm

if sys.platform.startswith('win'):
    import pandas as pd
else:
    import modin.pandas as pd
from src.snorkel.append_new_data import parallelize_dataframe
from src.snorkel.classification_based_weak_labelling_fv import classify_labeler_factory
from src.snorkel.weak_labellers import *


def main(argv):
    location = argv[0]
    languages = argv[1:]
    if len(languages) == 0:
        print("You should provide a list of languages to process", file=sys.stderr)

    df_train = pd.read_csv(location)

    for language in tqdm(languages, desc='Languages'):
        try:
            h5f = h5py.File('./data/frequency_data/%s/data_votes.h5' % language, 'r')
            L_train_existing = h5f['%s_votes' % language][:]
        finally:
            if 'h5f' in locals().keys():
                h5f.close()
        # clf_labeling_factory = classify_labeler_factory(language)
        lfs_tags = [
                       lf_bruteforce_tag_factory(language, tag_encoders)
                   ] + [
                       # clf_labeling_factory(n) for n in range(7) if n != 5  # Exclude Naive Bayes
                   ]
        tapplier = PandasLFApplier(lfs_tags)

        if sys.platform.startswith('win'):
            L_train = parallelize_dataframe(df_train, tapplier.apply, 6)
        else:
            # modin is not compatible with progress_apply
            L_train = tapplier.apply(df_train, progress_bar=False)

        L_train = np.c_[L_train_existing, L_train]

        try:
            os.makedirs('./data/frequency_data/%s' % language, exist_ok=True)
            h5f = h5py.File('./data/frequency_data/%s/data_votes.h5' % language, 'w')
            h5f.create_dataset('%s_votes' % language, data=L_train)
        finally:
            if 'h5f' in locals().keys():
                h5f.close()


if __name__ == '__main__':
    main(sys.argv[1:])
