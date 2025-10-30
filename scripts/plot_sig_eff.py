import argparse
import logging
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import rootutils
from scipy.interpolate import interp1d
from sklearn.metrics import roc_curve

root = rootutils.setup_root(
    search_from=__file__,
    pythonpath=True,
)

logging.basicConfig(level=logging.INFO)
log = logging.getLogger()

# plot defaults
plt.rcParams["xaxis.labellocation"] = "right"
plt.rcParams["yaxis.labellocation"] = "top"
plt.rcParams["legend.edgecolor"] = "1"
plt.rcParams["legend.framealpha"] = 0.0
plt.rcParams["axes.labelsize"] = "large"
plt.rcParams["axes.titlesize"] = "large"
plt.rcParams["legend.fontsize"] = 21


def get_signal_efficiency(
    labels: np.ndarray,
    predictions: np.ndarray,
    background_rejection: float,
) -> float:
    """Calculate the signal efficiency at a given background rejection."""
    fpr, tpr, _ = roc_curve(labels, predictions)  # Get the ROC curve
    # Interpolate to get the signal efficiency at the given background rejection
    target_fpr = 1 - background_rejection
    return interp1d(fpr, tpr, kind="linear", fill_value="extrapolate")(target_fpr)


def folder_split(folder_path: Path) -> tuple:
    """Split the folder name into the generation, dope, and seed."""
    gen = "herwig" if "herwig" in folder_path.name else "pythia"
    folder_path = folder_path.name.split(f"{gen}_")[1]
    dope, _, seed = folder_path.split("_")[:3]
    return (gen, int(dope), int(seed))


def parse_args():
    parser = argparse.ArgumentParser(description="Reshape, sort, and save data.")
    parser.add_argument(
        "--data_dir",
        type=Path,
        help="Directory containing the data files",
        default="/srv/beegfs/scratch/groups/rodem/LHCO/strong_cwola/lowstrongcwola_pipeline/",
    )
    parser.add_argument(
        "--output",
        type=Path,
        help="The output file to produce.",
        default="/srv/beegfs/scratch/groups/rodem/LHCO/strong_cwola/lowstrongcwola_pipeline/sic.png",
    )
    parser.add_argument(
        "--pattern",
        type=str,
        help="search string for folders",
        default="*_*_seed_*_combined",
    )
    parser.add_argument(
        "--ignore_pattern",
        type=str,
        help="search string for folders",
        default="3000",
    )
    # Add an argument for if is pretrained or not
    parser.add_argument(
        "--pretrained",
        action="store_true",
        help="If the model is pretrained or not",
    )
    # Add an argument for if it uses bdts or not
    parser.add_argument(
        "--is_bdt",
        action="store_true",
        help="If the model is pretrained or not",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    log.info(f"Searching for folders matching pattern: {args.pattern}")
    folders = list(Path(args.data_dir).glob(args.pattern))
    # Split into paths containg "un_pretrained" and those that don't
    if args.is_bdt:
        folders = [f for f in folders if "bdt_" in f.name]
    else:
        folders = [f for f in folders if "bdt_" not in f.name]
        if args.pretrained:
            folders = [f for f in folders if "un_pretrained" not in f.name]
        else:
            folders = [f for f in folders if "un_pretrained" in f.name]
    # Drop any folders whos lowest level contains the ignore pattern
    folders = [f for f in folders if args.ignore_pattern not in f.name]
    log.info(f"Found {len(folders)} folders")

    # Sort the folders alphabetically
    folders = sorted(folders, key=folder_split)

    # Concatenate each outputs pythia and herwig file
    pythia_files = [f / "pythia.h5" for f in folders]
    herwig_files = [f / "herwig.h5" for f in folders]
    dataframes = [
        pd.concat([pd.read_hdf(pf), pd.read_hdf(hf)])
        for pf, hf in zip(pythia_files, herwig_files)
    ]
    df0 = dataframes[0]
    # check frequency of inf in df0
    inf_count = 0
    for col in df0.columns:
        inf_count += np.isinf(df0[col]).sum()
    log.info(f"Number of inf values in first dataframe: {inf_count}")
    # check frequency of nan in df0
    nan_count = 0
    for col in df0.columns:
        nan_count += np.isnan(df0[col]).sum()
    log.info(f"Number of nan values in first dataframe: {nan_count}")


if __name__ == "__main__":
    main()
