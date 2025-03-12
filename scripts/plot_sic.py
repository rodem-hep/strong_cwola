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
    return interp1d(1 / fpr, sic, kind="linear", fill_value="extrapolate")(x_space)


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

    log.info("Calculating the SIC for pythia samples")
    x_space = 10 ** (np.linspace(1, 5, 500))
    test_sics = [
        get_sic(
            d["labels"][d["is_pythia"] == 1],
            d["outputs"][d["is_pythia"] == 1],
            x_space,
        )
        for d in dataframes
    ]

    # Put it all in one dataframe
    log.info("Combining SIC scores into single dataframe")
    combined = {}
    for i, folder in enumerate(folders):
        gen, dope, seed = folder_split(folder)
        combined[gen, dope, seed] = test_sics[i]
    combined = pd.DataFrame(combined).T

    # Group the seeds together, get the mean and std
    log.info("Grouping the seeds together")
    combined = combined.groupby(level=[0, 1]).agg(["mean", "std"])

    # colours - based on n_dope
    colours = {0: "k", 1000: "r", 3000: "g", 5000: "r"}

    # Plot the SIC
    log.info("Plotting the SIC")
    fig, axis = plt.subplots(1, 1, figsize=(8, 6))
    for gen, dope in combined.index:
        mean_sics = combined.loc[gen, dope][0::2].values
        mean_stds = combined.loc[gen, dope][1::2].values
        color = colours[dope] if gen != "herwig" else "b"
        label = f"{gen} {dope}"
        axis.plot(x_space, mean_sics, color, label=label)
        axis.fill_between(
            x_space,
            mean_sics - mean_stds,
            mean_sics + mean_stds,
            color=color,
            alpha=0.2,
        )
    axis.legend()
    axis.set_xscale("log")
    axis.set_xlim(x_space[0], x_space[-1])
    axis.set_ylim(bottom=0)
    axis.set_xlabel("1 / FPR")
    axis.set_ylabel("SIC")
    fig.savefig(args.output)
    plt.close()


if __name__ == "__main__":
    main()
