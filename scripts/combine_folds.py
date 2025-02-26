import argparse
import logging
from pathlib import Path

import rootutils

root = rootutils.setup_root(search_from=__file__, pythonpath=True)

from src.utils import combine_folds

logging.basicConfig(level=logging.INFO)
log = logging.getLogger()


def parse_args():
    parser = argparse.ArgumentParser(description="Reshape, sort, and save data.")
    parser.add_argument(
        "--data_dir",
        type=str,
        help="Directory containing the data files",
        default="/srv/beegfs/scratch/groups/rodem/LHCO/strong_cwola/lowstrongcwola/",
    )
    parser.add_argument(
        "--pattern",
        type=str,
        help="search string for folders",
        default="pythia_1000_seed_0_fold_*",
    )
    parser.add_argument(
        "--output_path",
        type=str,
        help="output path for the combined data",
        default="pythia_1000_seed_0_combined",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    log.info(f"Searching for folders matching pattern: {args.pattern}")
    folders = list(Path(args.data_dir).glob(args.pattern))
    log.info(f"Found {len(folders)} folders")

    # The output file names are based on the dataset used to train the classifier
    original_name = "pythia" if "pythia" in args.pattern else "herwig"
    additional_name = "herwig" if "pythia" in args.pattern else "pythia"

    log.info("Stacking folds from the original test data")
    test_files = [f / "original_test.h5" for f in folders]
    out_file = Path(args.data_dir) / args.output_path / (original_name + ".h5")
    combine_folds(test_files, out_file)

    log.info("Stacking folds from the additional test data")
    test_files = [f / "additional_test.h5" for f in folders]
    out_file = Path(args.data_dir) / args.output_path / (additional_name + ".h5")
    combine_folds(test_files, out_file, is_add=True)


if __name__ == "__main__":
    main()
