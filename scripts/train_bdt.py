# Use BDTs to run the classification of the data and KDEs to estimate the mass profiles

import argparse
import logging
import random
from pathlib import Path

import numpy as np
import pandas as pd
import rootutils
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.preprocessing import MinMaxScaler
from tqdm import trange

root = rootutils.setup_root(search_from=__file__, pythonpath=True)

from src.data.utils import get_hlf, load_dijet_file, load_strong_cwola_data

log = logging.getLogger(__name__)


def get_high_level(data: np.array) -> np.ndarray:
    # Drop mjj and split into two distinct jets
    j1, j2 = np.split(data[:, :-1], 2, axis=1)
    # Convert to the high level features and recombine
    return np.concatenate((get_hlf(j1), get_hlf(j2)), axis=1)


def k_fold_split(dataset: np.ndarray, num_folds: int, fold_idx: int) -> tuple:
    """Perform a k-fold splitting of a numpy array."""
    # Check the settings
    assert num_folds > 0
    assert fold_idx < num_folds

    # Get the fold index of each element in the set
    in_k = np.arange(len(dataset)) % num_folds

    # Get the indicies of the test val and train sets
    test_fold = fold_idx
    train_folds = [i for i in range(num_folds) if i != test_fold]

    # Get a mask to filter the dataset into train val and test
    in_test = in_k == test_fold
    in_train = np.isin(in_k, train_folds)

    # Use the masks to split the datasets
    test = dataset[in_test]
    train = dataset[in_train]

    return train, test


def get_ensemble_preditions(
    x_train: np.ndarray,
    y_train: np.ndarray,
    x_test: np.ndarray,
    add_test: np.ndarray,
    num_ensemble: int,
    random_state: int,
) -> np.ndarray:
    """Train an ensemble of classifiers and return the predictions on the test set."""
    preds = []
    add_test_preds = []
    for j in trange(num_ensemble, leave=False):
        # Create the decision tree
        clf = HistGradientBoostingClassifier(
            max_iter=200,
            early_stopping=True,
            validation_fraction=0.25,  # Matches k folding set up
            random_state=j + random_state,  # Results in a new train/val split
        )

        # Fit the classifier
        clf.fit(x_train, y_train)

        # Only collect the predictions for the signal class
        preds.append(clf.predict_proba(x_test)[:, 1])
        add_test_preds.append(clf.predict_proba(add_test)[:, 1])

    # Average the predictions of the ensemble
    return np.mean(preds, axis=0), np.mean(add_test_preds, axis=0)


def main(cfg) -> None:
    # Set the seed for reproducibility and enforcing variability
    log.info(f"Setting the seed to {cfg.seed}")
    np.random.seed(cfg.seed)
    random.seed(cfg.seed)

    # Load the data
    log.info("Loading data")
    data_dict = load_strong_cwola_data(
        cfg.background,
        cfg.signal,
        mjj_window=[2900, 4100],
        n_sig=cfg.n_sig,
        n_bkg=cfg.n_bkg,
        n_dope=cfg.n_dope,
        n_csts=None,
    )
    add_test_dict = load_dijet_file(
        cfg.add_test,
        mjj_window=[2900, 4100],
        n_events=cfg.n_bkg,
        n_csts=None,
    )
    # Grab the relevant components
    data = get_high_level(data_dict["ctxt"])
    labels = data_dict["cwola_labels"]
    true_labels = data_dict["labels"]
    add_test = get_high_level(add_test_dict["ctxt"])

    # Set up the classification
    log.info("Setting up the classification")
    # Shuffle the data and labels together
    idx = np.random.permutation(len(data))
    data = data[idx]
    labels = labels[idx]
    true_labels = true_labels[idx]
    # Preprocess everything
    scaler = MinMaxScaler()
    data = scaler.fit_transform(data)
    add_test = scaler.transform(add_test)
    # Make an array to fill with predictions
    predictions = np.zeros(len(data))
    add_predictions = np.zeros((len(add_test), cfg.num_folds))
    for fold_idx in range(cfg.num_folds):
        log.info(f"Starting fold {fold_idx}")

        # Split the data from the signal region using the fold_idx
        indx_train, indx_test = k_fold_split(
            np.arange(len(data)), cfg.num_folds, fold_idx
        )

        # Train an ensemble of decision trees
        log.info("- training ensemble")
        preds, add_preds = get_ensemble_preditions(
            data[indx_train],
            labels[indx_train],
            data[indx_test],
            add_test,
            cfg.num_ensemble,
            cfg.seed,
        )

        # Add the predictions to the array
        predictions[indx_test] = preds
        add_predictions[:, fold_idx] = add_preds

    # Split out background and signal predictions
    bkg_preds = predictions[true_labels == 0]
    sig_preds = predictions[true_labels == 1]
    # Pick one of the folds randomly to assign a score to the additional test set
    add_preds = add_predictions[
        np.arange(len(add_predictions)),
        np.random.randint(cfg.num_folds, size=len(add_predictions)),
    ]

    # Build the pythia and herwig predictions
    if cfg.is_herwig:
        pythia_preds = add_preds
        herwig_preds = bkg_preds
    else:
        pythia_preds = bkg_preds
        herwig_preds = add_preds
    predictions = np.concatenate((pythia_preds, sig_preds))
    sv_labels = np.concatenate((np.zeros(len(pythia_preds)), np.ones(len(sig_preds))))
    log.info("Saving the data")

    # Save as the dataframe with labels, predictions and is_pythia columns
    pd.DataFrame({
        "outputs": predictions,
        "labels": sv_labels,
        "is_pythia": np.ones(len(predictions)),
    }).to_hdf(cfg.output_dir / "pythia.h5", key="data", mode="w")
    # Save the additional test set
    pd.DataFrame({
        "outputs": herwig_preds,
        "labels": np.zeros(len(herwig_preds)),
        "is_pythia": np.zeros(len(herwig_preds)),
    }).to_hdf(cfg.output_dir / "herwig.h5", key="data", mode="w")

    log.info("All done.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Run BDT classification and KDE estimation."
    )
    parser.add_argument(
        "--background",
        type=Path,
        help="Path to what is treated as background.",
    )
    parser.add_argument(
        "--signal",
        type=Path,
        help="Path to what is treated as signal.",
    )
    parser.add_argument(
        "--add_test",
        type=Path,
        help="Path to additional test set.",
    )
    parser.add_argument(
        "--seed", type=int, default=42, help="Seed for random number generation"
    )
    parser.add_argument(
        "--num_folds", type=int, default=5, help="Number of folds for k-fold splitting"
    )
    parser.add_argument(
        "--num_ensemble", type=int, required=True, help="Number of ensemble classifiers"
    )
    parser.add_argument(
        "--output_dir",
        type=Path,
        help="Where to save predictions on background and signal.",
    )
    parser.add_argument(
        "--n_sig", type=int, help="Number of signal to use when doping."
    )
    parser.add_argument(
        "--n_bkg", type=int, help="Number of background to use when doping."
    )
    parser.add_argument(
        "--n_dope", type=int, help="Number of signal to use when doping."
    )
    parser.add_argument(
        "--is_herwig",
        action="store_true",
        help="Flag to indicate if Herwig data is used.",
    )
    args = parser.parse_args()
    main(args)
