import sys

import pandas as pd
from snorkel.labeling import LabelModel
from snorkel.labeling import PandasLFApplier

from src.preprocessor.so_to_pandas import SO_to_pandas
from src.snorkel.weak_labellers import frequency_labelling_function_factory


def main(argv):
    language = argv[0]
    location = argv[1]
    # Define the set of labeling functions (LFs)
    lfs = [frequency_labelling_function_factory(language)]
    # TODO: Get Cardinality as a function of language (build vocab per lang)
    size_tag_voc = 1000

    df_train = SO_to_pandas(location)

    # Apply the LFs to the unlabeled training data
    applier = PandasLFApplier(lfs)
    L_train = applier.apply(df_train)

    # Train the label model and compute the training labels
    label_model = LabelModel(cardinality=size_tag_voc, verbose=True)
    label_model.fit(L_train, n_epochs=2000, log_freq=200, seed=42)
    df_train["label"] = label_model.predict(L=L_train, tie_break_policy="abstain")


if __name__ == '__main__':
    main(sys.argv[1:])
