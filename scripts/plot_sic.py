import argparse
import logging
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import rootutils
from matplotlib import ticker
from scipy.interpolate import interp1d
from sklearn.metrics import roc_curve

root = rootutils.setup_root(search_from=__file__, pythonpath=True)


logging.basicConfig(level=logging.INFO)
log = logging.getLogger()

# Some defaults for my plots to make them look nicer
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

    log.info("Calculating the SIC for pythia samples")
    x_space = 10 ** (np.linspace(1, 4, 500))
    test_sics = [
        get_sic(
            d["labels"][d["is_pythia"] == 1],
            d["outputs"][d["is_pythia"] == 1],
            x_space,
        )
        for d in dataframes
    ]

    log.info("Calculating the signal efficiencies at 99% background rejection")
    test_sig_effs = [
        get_signal_efficiency(
            d["labels"][d["is_pythia"] == 1],
            d["outputs"][d["is_pythia"] == 1],
            0.99,
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

    # Calculate mean and std for each configuration
    sig_eff_stats = {}
    for key, values in sig_eff_mapping.items():
        sig_eff_stats[key] = {
            "mean": np.mean(values),
            "std": np.std(values),
            "values": values,
        }
    # write stats to text file
    with open(args.output.parent / f"{args.output.stem}_sig_eff_stats.txt", "w") as f:
        f.write("Signal efficiency at 99% background rejection:\n")
        for (gen, dope), stats in sig_eff_stats.items():
            f.write(f"{gen} {dope}: {stats['mean']:.4f} ± {stats['std']:.4f}\n")
        f.write("\nSignal efficiency gain relative to herwig 0:\n")
        herwig_0_mean = sig_eff_stats["herwig", 0]["mean"]
        for (gen, dope), stats in sig_eff_stats.items():
            gain = stats["mean"] / herwig_0_mean - 1
            f.write(f"{gen} {dope}: {gain * 100:.2f} %\n")

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
    # Set the style
    plt.style.use("tableau-colorblind10")
    # Get the first two colors from the color cycle
    dfc = plt.rcParams["axes.prop_cycle"].by_key()["color"]
    colours = {0: "k", 1700: dfc[0], 5_000: dfc[1], 10_000: dfc[2]}
    # linestyle - based on generator
    linestyle = {"herwig": "dashed", "pythia": "solid"}
    # Define a label map
    label_map = {
        "herwig 0": "Simulation proxy",
        "pythia 0": "sCWoLa 0",
        "pythia 1700": "sCWoLa 1700",
        "pythia 5000": "sCWoLa 5000",
    }

    # Increase the font size
    plt.rcParams.update({"font.size": 18})

    # Plot the SIC
    log.info("Plotting the SIC")
    fig, (axis, ratio_axis) = plt.subplots(
        2, 1, figsize=(8, 12), gridspec_kw={"height_ratios": [3, 1]}
    )
    pythia_0_sics = combined.loc["pythia", 0][0::2].values
    pythia_0_stds = combined.loc["pythia", 0][1::2].values

    for gen, dope in combined.index:
        mean_sics = combined.loc[gen, dope][0::2].values
        mean_stds = combined.loc[gen, dope][1::2].values
        color = colours[dope]
        # Capitalise the generator name
        axis.plot(
            x_space,
            mean_sics,
            color,
            label=label_map[f"{gen} {dope}"],
            linestyle=linestyle[gen],
        )
        axis.fill_between(
            x_space,
            mean_sics - mean_stds,
            mean_sics + mean_stds,
            color=color,
            alpha=0.2,
        )

        # Calculate the ratio and plot it
        ratio = mean_sics / pythia_0_sics
        ratio_uncertainty = ratio * np.sqrt(
            (mean_stds / mean_sics) ** 2 + (pythia_0_stds / pythia_0_sics) ** 2
        )
        ratio_axis.plot(
            x_space,
            ratio,
            color,
            linestyle=linestyle[gen],
        )
        ratio_axis.fill_between(
            x_space,
            ratio - ratio_uncertainty,
            ratio + ratio_uncertainty,
            color=color,
            alpha=0.2,
        )

    # Plot the legend in the top left corner
    axis.legend(frameon=False, loc="upper left")
    axis.set_xscale("log")
    axis.set_xlim(x_space[0], x_space[-1])
    axis.set_ylim(0, 60)
    axis.set_xlabel(r"Rejection $(1/\epsilon_b)$")
    axis.set_ylabel("Significance improvement (SIC)")
    # Add a grid with x-axis points only every 10^i
    axis.set_xticks([10**i for i in range(1, 5)])
    axis.xaxis.set_major_locator(ticker.LogLocator(base=10.0, numticks=5))
    axis.xaxis.set_minor_locator(ticker.NullLocator())
    axis.grid(True, which="both", linestyle="--", alpha=0.5)

    # Configure the ratio plot
    ratio_axis.set_xscale("log")
    ratio_axis.set_xlim(x_space[0], x_space[-1])
    ratio_axis.set_ylim(0.8, 1.1)
    ratio_axis.set_xlabel(r"Rejection $(1/\epsilon_b)$")
    ratio_axis.set_ylabel(f"Ratio to\n{label_map['pythia 0']}")
    ratio_axis.set_xticks([10**i for i in range(1, 5)])
    ratio_axis.xaxis.set_major_locator(ticker.LogLocator(base=10.0, numticks=5))
    ratio_axis.xaxis.set_minor_locator(ticker.NullLocator())
    ratio_axis.grid(True, which="both", linestyle="--", alpha=0.5)

    # Save the figure
    fig.savefig(args.output, bbox_inches="tight")
    plt.close()

    # repeat plotting but with ratio to herwig 0
    log.info("Plotting the SIC with ratio to herwig 0")
    fig, (axis, ratio_axis) = plt.subplots(
        2, 1, figsize=(8, 12), gridspec_kw={"height_ratios": [3, 1]}
    )
    herwig_0_sics = combined.loc["herwig", 0][0::2].values
    herwig_0_stds = combined.loc["herwig", 0][1::2].values

    for gen, dope in combined.index:
        mean_sics = combined.loc[gen, dope][0::2].values
        mean_stds = combined.loc[gen, dope][1::2].values
        color = colours[dope]
        # Capitalise the generator name
        axis.plot(
            x_space,
            mean_sics,
            color,
            label=label_map[f"{gen} {dope}"],
            linestyle=linestyle[gen],
        )
        axis.fill_between(
            x_space,
            mean_sics - mean_stds,
            mean_sics + mean_stds,
            color=color,
            alpha=0.2,
        )

        # Calculate the ratio and plot it
        ratio = mean_sics / herwig_0_sics
        ratio_uncertainty = ratio * np.sqrt(
            (mean_stds / mean_sics) ** 2 + (herwig_0_stds / herwig_0_sics) ** 2
        )
        ratio_axis.plot(
            x_space,
            ratio,
            color,
            linestyle=linestyle[gen],
        )
        ratio_axis.fill_between(
            x_space,
            ratio - ratio_uncertainty,
            ratio + ratio_uncertainty,
            color=color,
            alpha=0.2,
        )

    # Plot the legend in the top left corner
    axis.legend(frameon=False, loc="upper left")
    axis.set_xscale("log")
    axis.set_xlim(x_space[0], x_space[-1])
    axis.set_ylim(0, 60)
    axis.set_xlabel(r"Rejection $(1/\epsilon_b)$")
    axis.set_ylabel("Significance improvement (SIC)")
    # Add a grid with x-axis points only every 10^i
    axis.set_xticks([10**i for i in range(1, 5)])
    axis.xaxis.set_major_locator(ticker.LogLocator(base=10.0, numticks=5))
    axis.xaxis.set_minor_locator(ticker.NullLocator())
    axis.grid(True, which="both", linestyle="--", alpha=0.5)
    # Configure the ratio plot
    ratio_axis.set_xscale("log")
    ratio_axis.set_xlim(x_space[0], x_space[-1])
    ratio_axis.set_ylim(0.8, 1.2)
    ratio_axis.set_xlabel(r"Rejection $(1/\epsilon_b)$")
    ratio_axis.set_ylabel(f"Ratio to\n{label_map['herwig 0']}")
    ratio_axis.set_xticks([10**i for i in range(1, 5)])
    ratio_axis.xaxis.set_major_locator(ticker.LogLocator(base=10.0, numticks=5))
    ratio_axis.xaxis.set_minor_locator(ticker.NullLocator())
    ratio_axis.grid(True, which="both", linestyle="--", alpha=0.5)
    # Save the figure
    herwig_output = (
        args.output.parent / f"{args.output.stem}_H_ratio{args.output.suffix}"
    )
    fig.savefig(herwig_output, bbox_inches="tight")
    plt.close()


if __name__ == "__main__":
    main()
