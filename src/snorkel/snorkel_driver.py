from snorkel.labeling import PandasLFApplier
from snorkel.labeling.model import LabelModel


def main():
    # Define the set of labeling functions (LFs)
    lfs = [lf_keyword_my, lf_regex_check_out, lf_short_comment, lf_textblob_polarity]

    # Apply the LFs to the unlabeled training data
    applier = PandasLFApplier(lfs)
    L_train = applier.apply(df_train)

    # Train the label model and compute the training labels
    label_model = LabelModel(cardinality=2, verbose=True)
    label_model.fit(L_train, n_epochs=500, log_freq=50, seed=123)
    df_train["label"] = label_model.predict(L=L_train, tie_break_policy="abstain")


if __name__ == '__main__':
    main()
