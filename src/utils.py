from copy import deepcopy
from pathlib import Path

import h5py
import numpy as np
import pandas as pd


def combine_folds(
    file_list: list,
    output_path: Path,
    is_add: bool = False,
) -> None:
    """Combine and save the exported files from the classifier training."""
    data = []  # Load the data from all files
    for file_path in file_list:
        with h5py.File(file_path, "r") as f:
            data.append({key: f[key][:] for key in f})

    # Check if we are adding the samples
    # These samples should be ordered in the exact same way
    if is_add:
        assert len({len(d["outputs"]) for d in data}) == 1  # All files same length
        stacked = deepcopy(data[0])  # Pull out first one to be the reference
        rng = np.random.default_rng()
        idxes = rng.integers(0, len(data), len(stacked["outputs"]))  # Randomize order
        for i, idx in enumerate(idxes):
            stacked["outputs"][i] = data[idx]["outputs"][i]  # Choose a random sample

    # Otherwise we concatenate the samples from independant folds together
    else:
        stacked = {key: np.vstack([d[key] for d in data]) for key in data[0]}

    # Save the data to a new file readable by pandas
    output_path.parent.mkdir(parents=True, exist_ok=True)
    stacked = {k: v.flatten() for k, v in stacked.items()}  # Pandas needs flat arrays
    stacked = pd.DataFrame(stacked)
    stacked.to_hdf(output_path, key="data", mode="w")
