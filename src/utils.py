from copy import deepcopy
from pathlib import Path

import h5py
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.interpolate import interp1d
from sklearn.metrics import roc_curve


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


def folder_split(folder_path: Path) -> tuple:
    """Split the folder name into the generation, dope, and seed."""
    gen = "herwig" if "herwig" in folder_path.name else "pythia"
    folder_path = folder_path.name.split(f"{gen}_")[1]
    dope, _, seed = folder_path.split("_")[:3]
    return (gen, int(dope), int(seed))


def get_signal_efficiency(
    labels: np.ndarray,
    predictions: np.ndarray,
    background_rejection: float | list[float],
) -> float | list[float]:
    """Calculate the signal efficiency at a given background rejection."""
    fpr, tpr, _ = roc_curve(labels, predictions)  # Get the ROC curve
    interpolator = interp1d(fpr, tpr, kind="linear", fill_value="extrapolate")

    # Handle both single and multiple background rejection values
    if isinstance(background_rejection, list):
        target_fprs = [1 - br for br in background_rejection]
        return [float(interpolator(target_fpr)) for target_fpr in target_fprs]

    target_fpr = 1 - background_rejection
    return float(interpolator(target_fpr))


def plot_signal_efficiency_gain(
    dataframes: pd.DataFrame,
    folders: list[str],
    bin_centers: list[float],
    background_rejections: list[float],
    feature: str,
    output: str,
) -> None:
    """Calculate binned signal efficiency and gain for a given feature and produce plots
    at different background rejections.
    """
    sig_eff_by_feat = {}
    for br in background_rejections:
        for i, folder in enumerate(folders):
            gen, dope, _ = folder_split(folder)
            df = dataframes[i]
            for feat_bin in range(len(bin_centers[feature])):
                bin_df = df[df[f"{feature}_bin"] == feat_bin]
                try:
                    sig_eff = get_signal_efficiency(
                        bin_df["labels"][bin_df["is_pythia"] == 1],
                        bin_df["outputs"][bin_df["is_pythia"] == 1],
                        br,
                    )
                except:
                    print("sig_eff calculation failed")
                    breakpoint()
                if (gen, dope, feat_bin, br) not in sig_eff_by_feat:
                    sig_eff_by_feat[gen, dope, feat_bin, br] = []
                sig_eff_by_feat[gen, dope, feat_bin, br].append(sig_eff)

    combined = pd.DataFrame(sig_eff_by_feat).T
    combined["mean"] = combined.mean(axis=1)
    combined["std"] = combined.std(axis=1)

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

    feature_name_unit_map = {
        "mjj": ["m_{{jj}}", "GeV"],
        "pt_1": ["p_{T}^{1}", ""],
        "pt_2": ["p_{T}^{2}", ""],
        "tau_21_1": [r"\tau_{{21}}^{{1}}", ""],
        "tau_21_2": [r"\tau_{{21}}^{{2}}", ""],
        "tau_32_1": [r"\tau_{{32}}^{{1}}", ""],
        "tau_32_2": [r"\tau_{{32}}^{{2}}", ""],
    }
    # Increase the font size
    plt.rcParams.update({"font.size": 18})

    # plot signal efficiency vs mjj bin center

    plt.figure(figsize=(8, 6))
    for br in background_rejections:
        combined_br = combined.xs(br, level=3)
        for (gen, dope), group in combined_br.groupby(level=[0, 1]):
            plt.errorbar(
                bin_centers[feature],
                group["mean"],
                yerr=group["std"],
                label=label_map.get(f"{gen} {dope}", f"{gen} {dope}"),
                color=colours.get(dope, "gray"),
                linestyle=linestyle.get(gen, "solid"),
                marker="o",
                capsize=5,
            )
        plt.xlabel(
            rf"${feature_name_unit_map[feature][0]}$ {feature_name_unit_map[feature][1]}"
        )
        plt.title(f"Signal Efficiency at {br * 100}% Background Rejection")
        plt.ylabel("Signal Efficiency")
        plt.legend()
        plt.grid(True, which="both", ls="--", lw=0.5)
        plt.tight_layout()
        # remove .pdf suffix and replae with _test.pdf
        output_path = output.parent / f"{output.stem}_sig_eff_{feature}_{br}.pdf"
        plt.savefig(output_path, bbox_inches="tight")
        plt.close()

    # plot gain in signal efficiency relative to herwig 0 vs mjj bin center
    plt.figure(figsize=(8, 6))
    herwig_0 = combined.xs(("herwig", 0))
    for br in background_rejections:
        herwig_0_br = herwig_0.xs(br, level=1)
        combined_br = combined.xs(br, level=3)
        for (gen, dope), group in combined_br.groupby(level=[0, 1]):
            gain_mean = []
            gain_std = []
            for bin_idx in range(len(bin_centers[feature])):
                if (gen, dope) == ("herwig", 0):
                    gain_mean.append(0)
                    gain_std.append(0)
                    continue
                gain_mean.append(
                    group["mean"][bin_idx] / herwig_0_br["mean"][bin_idx] - 1
                )
                gain_std.append(
                    np.sqrt(
                        (group["std"][bin_idx] / herwig_0_br["mean"][bin_idx]) ** 2
                        + (
                            group["mean"][bin_idx]
                            * herwig_0_br["std"][bin_idx]
                            / herwig_0_br["mean"][bin_idx] ** 2
                        )
                        ** 2
                    )
                )
            plt.errorbar(
                bin_centers[feature],
                gain_mean,
                yerr=gain_std,
                label=label_map.get(f"{gen} {dope}", f"{gen} {dope}"),
                color=colours.get(dope, "gray"),
                linestyle=linestyle.get(gen, "solid"),
                marker="o",
                capsize=5,
            )
        plt.xlabel(
            rf"${feature_name_unit_map[feature][0]}$ {feature_name_unit_map[feature][1]}"
        )
        plt.title(f"Signal Efficiency Gain at {br * 100}% Background Rejection")
        plt.ylabel("Signal Efficiency Gain")
        plt.legend()
        plt.grid(True, which="both", ls="--", lw=0.5)
        plt.tight_layout()
        output_path = output.parent / f"{output.stem}_sig_eff_gain_{feature}_{br}.pdf"
        plt.savefig(output_path, bbox_inches="tight")
        plt.close()
