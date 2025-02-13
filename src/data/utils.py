import logging

import h5py
import numpy as np

log = logging.getLogger(__name__)


def k_fold_split(dataset: dict, num_folds: int, test_fold: int) -> tuple:
    """Perform a k-fold splitting of a mappable dataset."""
    assert num_folds > 0, "The number of folds must be greater than 0"
    assert test_fold < num_folds - 1, "The test fold index must be less than num_folds"

    # Get the fold index of each element in the dataset
    n = next(iter(dataset.values())).shape[0]
    in_k = np.arange(n) % num_folds  # The fold to put each element in

    # Get the indicies of the test, val and train folds based on th current permutation
    val_fold = (test_fold + 1) % num_folds  # Val is the next fold
    train_folds = [i for i in range(num_folds) if i not in {test_fold, val_fold}]

    # Get a mask to filter the dataset into train val and test
    in_test = in_k == test_fold
    in_val = in_k == val_fold
    in_train = np.isin(in_k, train_folds)

    # Use the masks to split the datasets
    test = {k: v[in_test] for k, v in dataset.items()}
    valid = {k: v[in_val] for k, v in dataset.items()}
    train = {k: v[in_train] for k, v in dataset.items()}

    return train, valid, test


def get_mass_mask(x: np.ndarray, windows: tuple | None) -> np.ndarray:
    """Create a mask based on whether a value falls within a window."""
    if windows is None:  # No cut, return all
        return np.ones(x.shape, dtype=bool)
    if np.ndim(windows) == 1:  # Single window, make it a tuple for consistency
        windows = (windows,)
    mask = np.zeros(x.shape, dtype=bool)
    for min_, max_ in windows:  # Check if value it is in at least one
        mask = mask | ((min_ <= x) & (x < max_))
    return mask


def load_dijet_file(
    file_path: str,
    mjj_window: tuple | list | None = None,
    n_events: int | None = None,
    n_csts: int | None = None,
    from_bottom: bool = False,
) -> dict:
    """Load in information from a diject file."""
    data = {}
    with h5py.File(file_path, "r") as f:
        for key in f:
            if from_bottom and len(f[key].shape) > 1:
                data[key] = f[key][-n_events:, :n_csts].astype(np.float32)
            elif from_bottom:
                data[key] = f[key][-n_events:].astype(np.float32)
            elif len(f[key].shape) > 1:
                data[key] = f[key][:n_events, :n_csts].astype(np.float32)
            else:
                data[key] = f[key][:n_events].astype(np.float32)
    mask = get_mass_mask(data["mjj"], mjj_window).reshape(-1)
    return {k: v[mask] for k, v in data.items()}


def load_strong_cwola_data(
    bkg_path: str,
    sig_path: str,
    mjj_window: tuple | list | None = None,
    n_sig: int | None = None,
    n_bkg: int | None = None,
    n_dop: int | None = None,
    n_csts: int | None = 0,
) -> tuple:
    bkg_data = load_dijet_file(bkg_path, mjj_window, n_bkg, n_csts)
    dop_data = load_dijet_file(sig_path, mjj_window, n_dop, n_csts, from_bottom=True)
    sig_data = load_dijet_file(sig_path, mjj_window, n_sig, n_csts)
    bkg_data["cwola_labels"] = np.zeros_like(bkg_data["labels"])
    dop_data["cwola_labels"] = np.zeros_like(dop_data["labels"])
    sig_data["cwola_labels"] = np.ones_like(sig_data["labels"])
    return {
        k: np.concat([bkg_data[k], dop_data[k], sig_data[k]], axis=0) for k in bkg_data
    }
