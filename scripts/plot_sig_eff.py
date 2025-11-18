import argparse
import logging
from datetime import datetime
from pathlib import Path

import h5py
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

from src.utils import (
    folder_split,
    get_signal_efficiency,
    plot_signal_efficiency_gain,
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


def background_rejection_to_fpr(
    background_rejection: list[float],
) -> list[float]:
    """Convert background rejection to false positive rate (FPR)."""
    return [1 - br for br in background_rejection]


def load_and_cast(path):
    """Ensure all numerical columns are float64."""
    df = pd.read_hdf(path)
    for col in df.select_dtypes(include=[np.number]).columns:
        df[col] = df[col].astype(np.float64)
    return df


def merge_jets(
    clustered_sig: Path,
    clustered_bkg: Path,
) -> pd.DataFrame:
    # get event_ids
    files = [clustered_sig, clustered_bkg]
    event_ids = []
    jets1 = []
    jets2 = []
    for file in files:
        with h5py.File(file, "r") as f:
            event_ids.append(f["event_ids"][:].astype(np.int32))
            jets1.append(f["jets1"][:].astype(np.float32))
            jets2.append(f["jets2"][:].astype(np.float32))
    event_ids = np.concatenate(event_ids).flatten()
    jets1 = np.concatenate(jets1)
    jets2 = np.concatenate(jets2)

    jet_features = ["pt", "eta", "phi", "m", "tau_1", "tau_2", "tau_3"]

    # Create DataFrame dictionary with a loop
    data_dict = {"event_ids": event_ids}

    for i, jet_array in enumerate([jets1, jets2], start=1):
        for j, feature in enumerate(jet_features):
            data_dict[f"{feature}_{i}"] = jet_array[:, j]

    return pd.DataFrame(data_dict)


def match_event_ids(
    output_df: pd.DataFrame,
    jet_df: pd.DataFrame,
) -> pd.DataFrame:
    """Merge jet_df and output_df on event_ids, returning only rows with event_ids in output_df."""
    return pd.merge(output_df, jet_df, on="event_ids", how="left")


def match_dataframes(
    output_dfs: list[pd.DataFrame], jet_df: pd.DataFrame
) -> pd.DataFrame:
    """Merge list of output_dfs on event_ids."""
    return [match_event_ids(output_df, jet_df) for output_df in output_dfs]


def feature_bin(
    combined_dfs: list[pd.DataFrame],
    features: list[str],
    n_bins: int = 3,
    quantile_cut: bool = False,
) -> tuple[list[pd.DataFrame], dict[str, list[float]]]:
    """Bin the combined dataframe list in a number of equally populated bins.
    Bins are computed based on is_pythia == 1 samples only.
    """
    centers_dict = {}
    for feature in features:
        for df in combined_dfs:
            # Compute bins based only on is_pythia == 1 samples
            pythia_mask = df["is_pythia"] == 1
            if quantile_cut:
                _, bins = pd.qcut(
                    df.loc[pythia_mask, feature],
                    q=n_bins,
                    labels=False,
                    duplicates="drop",
                    retbins=True,
                )
            else:
                _, bins = pd.cut(
                    df.loc[pythia_mask, feature],
                    bins=n_bins,
                    labels=False,
                    retbins=True,
                )

            # Apply bins to all samples using pd.cut
            df[f"{feature}_bin"] = pd.cut(
                df[feature], bins=bins, labels=False, include_lowest=True
            )

        centers_dict[feature] = [
            (bins[i] + bins[i + 1]) / 2 for i in range(len(bins) - 1)
        ]
    return combined_dfs, centers_dict


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
    base_data_dir = Path(args.data_dir).parent
    signal_path = base_data_dir / "clustered_pythia_sig.h5"
    background_path = base_data_dir / "clustered_pythia_bkg.h5"
    # load signal and background dataframes - can't use pd.read_hdf directly due to non-standard structure
    merged_jets = merge_jets(signal_path, background_path)
    merged_jets["tau_21_1"] = merged_jets["tau_2_1"] / merged_jets["tau_1_1"]
    merged_jets["tau_21_2"] = merged_jets["tau_2_2"] / merged_jets["tau_1_2"]
    merged_jets["tau_32_1"] = merged_jets["tau_3_1"] / merged_jets["tau_2_1"]
    merged_jets["tau_32_2"] = merged_jets["tau_3_2"] / merged_jets["tau_2_2"]
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
        pd.concat([load_and_cast(pf), load_and_cast(hf)])
        for pf, hf in zip(pythia_files, herwig_files)
    ]
    background_rejections = [0.90, 0.95, 0.99]
    log.info(
        f"Calculating the signal efficiencies at {background_rejections} background rejection"
    )
    test_sig_effs = [
        get_signal_efficiency(
            d["labels"][d["is_pythia"] == 1],
            d["outputs"][d["is_pythia"] == 1],
            background_rejections,
        )
        for d in dataframes
    ]

    # Create a mapping for signal efficiencies to identify configurations
    log.info("Creating signal efficiency mapping")
    sig_eff_mapping = {}
    for i, folder in enumerate(folders):
        gen, dope, seed = folder_split(folder)
        if (gen, dope) not in sig_eff_mapping:
            sig_eff_mapping[gen, dope] = []
        sig_eff_mapping[gen, dope].append(test_sig_effs[i])

    # Calculate mean and std for each configuration across seeds at each background rejection
    sig_eff_stats = {}
    for key, values in sig_eff_mapping.items():
        br_dict = {}
        for i, br in enumerate(background_rejections):
            if str(br) not in br_dict:
                br_dict[str(br)] = []
            br_dict[str(br)].extend([v[i] for v in values])
        # Now calculate mean and std for each background rejection
        sig_eff_stats[key] = {}
        for br_str, effs in br_dict.items():
            sig_eff_stats[key][br_str] = {
                "mean": np.mean(effs),
                "std": np.std(effs),
            }
    # write stats to text file for each background rejection
    for br in background_rejections:
        with open(
            args.output.parent / f"{args.output.stem}_sig_eff_stats_br_{br!s}.txt", "w"
        ) as f:
            f.write(f"Signal efficiency at {br!s}% background rejection:\n")
            for (gen, dope), stats in sig_eff_stats.items():
                f.write(
                    f"{gen} {dope}: {stats[str(br)]['mean']:.4f} ± {stats[str(br)]['std']:.4f}\n"
                )
            f.write("\nSignal efficiency gain relative to herwig 0:\n")
            herwig_0_mean = sig_eff_stats["herwig", 0][str(br)]["mean"]
            for (gen, dope), stats in sig_eff_stats.items():
                gain = stats[str(br)]["mean"] / herwig_0_mean - 1
                f.write(f"{gen} {dope}: {gain * 100:.2f} %\n")

    # get vars of interest
    merged_jets = merged_jets[
        ["event_ids", "pt_1", "pt_2", "tau_21_1", "tau_21_2", "tau_32_1", "tau_32_2"]
    ]
    # replace with matched datfame
    dataframes = match_dataframes(dataframes, merged_jets)

    # bin dataframes in 3 equal mjj bins
    log.info("Adding feature bin columns to dataframes")
    # features_of_interest=[
    #        "mjj",
    #        "pt_1",
    #        "pt_2",
    #        "tau_21_1",
    #        "tau_21_2",
    #        "tau_32_1",
    #        "tau_32_2",
    #    ],
    features_of_interest = [
        "mjj",
        "pt_1",
    ]
    dataframes, bin_centers = feature_bin(
        dataframes,
        features=features_of_interest,
        n_bins=3,
        quantile_cut=False,
    )
    plotting_features = features_of_interest
    log.info("Signal efficiency plotting")
    for feature in plotting_features:
        log.info(f"Plotting {feature}")
        plot_signal_efficiency_gain(
            dataframes,
            folders,
            bin_centers,
            background_rejections,
            feature=feature,
            output=args.output,
        )

    # write done .txt file
    Path(args.output).write_text(f"done at on {datetime.now()}\n")


if __name__ == "__main__":
    main()
