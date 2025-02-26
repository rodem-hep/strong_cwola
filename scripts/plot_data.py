import argparse
import logging
from pathlib import Path

import rootutils

root = rootutils.setup_root(search_from=__file__, pythonpath=True)

from mltools.mltools.plotting import plot_multi_hists
from src.data.utils import load_dijet_file

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
        default=1000,
    )
    parser.add_argument(
        "--n_csts",
        type=int,
        default=200,
        help="Maximum number of constituents per jet",
    )
    parser.add_argument(
        "--input_file",
        type=str,
        default="clustered_pythia_sig.h5",
        help="Path to the data file",
    )
    parser.add_argument(
        "--output_dir",
        type=Path,
        default="plots",
        help="Path to the data file",
    )
    parser.add_argument(
        "--data_dir",
        type=Path,
        default="/srv/beegfs/scratch/groups/rodem/LHCO/strong_cwola/",
        help="Path to the output file",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    input_file = args.data_dir / args.input_file
    output_dir = args.output_dir / input_file.stem
    output_dir.mkdir(parents=True, exist_ok=True)

    data = load_dijet_file(
        input_file,
        n_events=args.n_events,
        n_csts=args.n_csts,
    )

    # Constituents
    mask1 = data["csts1"][..., 0] > 0
    csts1 = data["csts1"][mask1]
    mask2 = data["csts2"][..., 0] > 0
    csts2 = data["csts2"][mask2]
    plot_multi_hists(
        [csts1, csts2],
        ["Constituents 1", "Constituents 2"],
        ["Pt", "Eta", "Phi"],
        path=output_dir / "constituents.png",
        logy=True,
    )

    # Jets
    plot_multi_hists(
        [data["jets1"], data["jets2"]],
        ["Jets 1", "Jets 2"],
        ["pt", "eta", "phi", "m", "tau1", "tau2", "tau3"],
        path=output_dir / "jets.png",
        logy=True,
    )

    # Dijet mass
    plot_multi_hists(
        [data["mjj"]],
        ["All events"],
        ["mjj"],
        path=output_dir / "mjj.png",
        logy=True,
    )


if __name__ == "__main__":
    main()
