import argparse
from pathlib import Path

import h5py
import numpy as np
import pandas as pd


def reshape_sort_save(data_dir: Path, raw_file: Path, out_file: Path) -> None:
    """Reshape the constituents, sort by Pt, and save to a new file."""
    print(f"Converting {raw_file} -> {out_file}")
    print(" - Loading data")
    csts = pd.read_hdf(data_dir / raw_file).to_numpy().astype(np.float32)
    has_labels = csts.shape[1] % 3 == 1
    if has_labels:
        labels = csts[:, -1].astype(bool)
        csts = csts[:, :-1]
    else:
        labels = np.zeros_like(csts[:, 0], dtype=bool)

    print(f" - Reshaping {csts.shape} -> ({csts.shape[0]}, -1, 3)")
    csts = csts.reshape(csts.shape[0], -1, 3)

    # Sort the constituents
    print(" - Sorting by Pt")
    order = np.argsort(-csts[..., 0], axis=-1)
    csts = np.take_along_axis(csts, order[..., None], axis=1)

    # Resave in a faster format for chunking
    print(" - Saving")
    with h5py.File(data_dir / out_file, "w") as file:
        file.create_dataset("csts", data=csts)
        file.create_dataset("labels", data=labels)


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
        "--out_file",
        type=Path,
        help="Name of the output file",
        default=Path("pythia.h5"),
    )
    return parser.parse_args()


def main():
    args = parse_args()
    reshape_sort_save(args.data_dir, args.raw_file, args.out_file)


if __name__ == "__main__":
    main()
