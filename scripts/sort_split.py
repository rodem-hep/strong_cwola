import argparse
from pathlib import Path

import h5py
import numpy as np
import pandas as pd


def parse_args():
    parser = argparse.ArgumentParser(description="Reshape, sort, and save data.")
    parser.add_argument(
        "--data_dir",
        type=Path,
        help="Directory containing the data files",
        default=Path("/srv/beegfs/scratch/groups/rodem/LHCO/strong_cwola/"),
    )
    parser.add_argument(
        "--raw_file",
        type=Path,
        help="Name of the raw data file",
        default=Path("events_anomalydetection_v2.h5"),
    )
    parser.add_argument(
        "--out_flag",
        type=str,
        help="Name of the output file",
        default="pythia",
    )
    return parser.parse_args()


def main():
    args = parse_args()

    print(f" - Loading data {args.raw_file}")
    csts = pd.read_hdf(args.data_dir / args.raw_file).to_numpy().astype(np.float32)
    has_labels = csts.shape[1] % 3 == 1
    if has_labels:
        labels = csts[:, -1].astype(bool)
        csts = csts[:, :-1]
    else:
        labels = np.zeros_like(csts[:, 0], dtype=bool)

    print(f" - Reshaping {csts.shape} -> ({csts.shape[0]}, -1, 3)")
    csts = csts.reshape(csts.shape[0], -1, 3)

    print(" - Sorting constituents by Pt")
    order = np.argsort(-csts[..., 0], axis=-1)
    csts = np.take_along_axis(csts, order[..., None], axis=1)

    # Check if there are signal events and save them separately
    sig_name = args.data_dir / (args.out_flag + "_sig.h5")
    print(f" - Saving signal to {sig_name}")
    with h5py.File(sig_name, "w") as f:
        f.create_dataset("csts", data=csts[labels])
        f.create_dataset("labels", data=labels[labels])

    # Check if there are background events and save them separately
    bkg_name = args.data_dir / (args.out_flag + "_bkg.h5")
    print(f" - Saving background to {bkg_name}")
    with h5py.File(bkg_name, "w") as f:
        f.create_dataset("csts", data=csts[~labels])
        f.create_dataset("labels", data=labels[~labels])


if __name__ == "__main__":
    main()
