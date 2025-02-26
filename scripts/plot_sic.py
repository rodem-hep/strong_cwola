import argparse
import logging
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import rootutils
from scipy.interpolate import interp1d
from sklearn.metrics import roc_curve

root = rootutils.setup_root(search_from=__file__, pythonpath=True)


logging.basicConfig(level=logging.INFO)
log = logging.getLogger()


def get_sic(
    labels: np.ndarray,
    predictions: np.ndarray,
    x_space: np.ndarray,
) -> tuple:
    fpr, tpr, _ = roc_curve(labels, predictions)  # Get the ROC curve
    tpr = tpr[fpr != 0]  # Mask away where the fpr is zero as it is used in the denom
    fpr = fpr[fpr != 0]

    # Calculate the SIC = tpr / sqrt(fpr)
    sic = tpr / np.sqrt(fpr)

    # For ease in plotting, interpolate on the fixed x_space
    return interp1d(tpr, sic, kind="linear", fill_value="extrapolate")(x_space)


def parse_args():
    parser = argparse.ArgumentParser(description="Reshape, sort, and save data.")
    parser.add_argument(
        "--data_dir",
        type=Path,
        help="Directory containing the data files",
        default="/srv/beegfs/scratch/groups/rodem/LHCO/strong_cwola/lowstrongcwola/",
    )
    parser.add_argument(
        "--pattern",
        type=str,
        help="search string for folders",
        default="*_*_seed_*_combined",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    log.info(f"Searching for folders matching pattern: {args.pattern}")
    folders = list(Path(args.data_dir).glob(args.pattern))
    log.info(f"Found {len(folders)} folders")

    # Sort the folders alphabetically
    folders = sorted(folders, key=lambda x: x.name)

    # Concatenate each outputs pythia and herwig file
    pythia_files = [f / "pythia.h5" for f in folders]
    herwig_files = [f / "herwig.h5" for f in folders]
    dataframes = [
        pd.concat([pd.read_hdf(pf), pd.read_hdf(hf)])
        for pf, hf in zip(pythia_files, herwig_files)
    ]

    log.info("Calculating the SIC for pythia samples")
    x_space = np.linspace(0.1, 1, 100)
    test_sics = [
        get_sic(
            d["labels"][d["is_pythia"] == 1],
            d["outputs"][d["is_pythia"] == 1],
            x_space,
        )
        for d in dataframes
    ]

    # Plot the SIC
    log.info("Plotting the SIC")
    fig, axis = plt.subplots(1, 1, figsize=(8, 6))
    for i, folder in enumerate(folders):
        gen, dope, _, seed, _ = folder.name.split("_")
        color = "b" if gen == "pythia" else "g"
        linestyle = "--" if dope == "1000" else "-"
        label = f"{gen} {dope}" if seed == "0" else None
        axis.plot(x_space, test_sics[i], color, linestyle=linestyle, label=label)
    axis.legend()
    axis.set_xlabel("Signal Efficiency")
    axis.set_ylabel("SIC")
    fig.savefig(args.data_dir / "sic.png")
    plt.close()


if __name__ == "__main__":
    main()
