import argparse
import logging
from pathlib import Path
import matplotlib.pyplot as plt
import numpy as np

import rootutils

root = rootutils.setup_root(search_from=__file__, pythonpath=True)

from mltools.mltools.plotting import plot_multi_hists
from src.data.utils import load_dijet_file, get_hlf

logging.basicConfig(level=logging.INFO)
log = logging.getLogger()


def none_or_int(x: str) -> int | None:
    if x.lower() == "none":
        return None
    return int(x)


def str_to_bool(x: str) -> bool:
    return x.lower() in {"t", "true", "y", "yes", "1"}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Cluster analysis parameters",
    )
    parser.add_argument(
        "--n_events",
        type=none_or_int,
        help="Number of events (None for all)",
        default=20_000,
    )
    parser.add_argument(
        "--n_csts",
        type=int,
        default=200,
        help="Maximum number of constituents per jet",
    )
    parser.add_argument(
        "--signal",
        type=Path,
        help="Path to the signal data file",
    )
    parser.add_argument(
        "--bkg",
        type=Path,
        help="Path to the background data file",
    )
    parser.add_argument(
        "--simulation",
        type=Path,
        help="Path to the simulation data file",
    )
    parser.add_argument(
        "--output",
        type=Path,
        help="Path to the data file",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    # Set the color cylce
    plt.style.use("tableau-colorblind10")
    # Load the data
    csts = {}
    j1_hl = {}
    j2_hl = {}
    for name, file in zip(
        ["Pythia", "Herwig", "Signal"], 
        [args.bkg, args.simulation, args.signal]
    ):
        data = load_dijet_file(
            file,
            n_events=args.n_events,
            n_csts=args.n_csts,
        )
        # Store the stacked constituents, with fake particles masked out
        csts[name] = data["csts"][data["csts"][..., 0] > 0][:, :-1]
        # Store the high level features but drop the mass
        hlf = data["ctxt"][:, :-1]
        # Split this into the two jets
        j1, j2 = np.split(hlf, 2, axis=1)
        # Store only the relevant high level features
        j1_hl[name] = get_hlf(j1)
        j2_hl[name] = get_hlf(j2)

    # # Constituents
    # plot_multi_hists(
    #     list(csts.values()),
    #     list(csts.keys()),
    #     [r"$p_{\mathrm{T}}$", r"$\Delta\eta$", r"$\Delta\phi$"],
    #     # ["Pt", "Eta", "Phi"],
    #     path=args.output,
    #     logy=True,
    #     bins=50,
    # )
    # High level features
    for i, (jet_nm, jet) in enumerate(zip(["jet1", "jet2"], [j1_hl, j2_hl])):
        plot_multi_hists(
            list(jet.values()),
            list(jet.keys()),
            [rf"$M_{i}$ [GeV]", rf"$\tau_{{21}}^{i}$", rf"$\tau_{{32}}^{i}$"],
            path=args.output.parent / f"{jet_nm}.pdf",
            logy=True,
            bins=50,
        )


if __name__ == "__main__":
    main()
