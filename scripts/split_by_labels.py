import argparse
from pathlib import Path

import h5py


def split_sig_bkg(data_dir: Path, input_file: Path) -> None:
    """Split a datafile into signal and background based on the labels."""
    with h5py.File(data_dir / input_file, "r") as f:
        data = {key: f[key][:] for key in f}

    # Create the mask based on the labels
    mask = data["labels"].astype(bool)

    # Split the data based on the mask and resave
    sig_name = data_dir / (input_file.stem + "_sig.h5")
    with h5py.File(sig_name, "w") as f:
        for key in data:
            f.create_dataset(key, data=data[key][mask])

    bkg_name = data_dir / (input_file.stem + "_bkg.h5")
    with h5py.File(bkg_name, "w") as f:
        for key in data:
            f.create_dataset(key, data=data[key][~mask])


def parse_args():
    parser = argparse.ArgumentParser(description="Split data by labels.")
    parser.add_argument(
        "--data_dir",
        type=Path,
        help="Directory containing the data files",
        default="/srv/beegfs/scratch/groups/rodem/LHCO/strong_cwola/",
    )
    parser.add_argument(
        "--input_file",
        type=Path,
        help="Name of the data file",
        default="clustered_pythia.h5",
    )
    return parser.parse_args()


def main():
    args = parse_args()
    split_sig_bkg(args.data_dir, args.input_file)


if __name__ == "__main__":
    main()
