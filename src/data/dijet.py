import logging
from functools import partial
from pathlib import Path

import numpy as np
from lightning import LightningDataModule
from torch.utils.data import DataLoader, Dataset

from mltools.mltools.torch_utils import train_valid_split
from src.data.preprocessing import collate_and_transform
from src.data.utils import k_fold_split, load_dijet_file, load_strong_cwola_data

log = logging.getLogger(__name__)


class DictDataset(Dataset):
    """Dataset that takes a dictionary of numpy arrays."""

    def __init__(self, data: dict) -> None:
        self.data = data

    def __len__(self) -> int:
        return len(next(iter(self.data.values())))

    def __getitem__(self, idx: int) -> dict:
        return {k: v[idx] for k, v in self.data.items()}


class PretrainingDataModule(LightningDataModule):
    """Datamodule for the pretraining where everything is loaded at once."""

    def __init__(
        self,
        *,
        data_dir: str,
        file_list: list,
        loader_kwargs: dict,
        mjj_window: tuple | list | None = None,
        n_csts: int | None = 0,
        val_frac: float = 0.1,
        transforms: list | None = None,
    ) -> None:
        super().__init__()
        self.loader_kwargs = loader_kwargs
        self.transforms = transforms

        # Load the full combined dataset
        file_list = [Path(data_dir, f) for f in file_list]
        data = [load_dijet_file(f, mjj_window, n_csts=n_csts) for f in file_list]
        data = {k: np.concat([f[k] for f in data], axis=0) for k in data[0]}
        data = DictDataset(data)

        # Split into train and valid
        self.train_set, self.valid_set = train_valid_split(data, val_frac)
        log.info(f"Train set size: {len(self.train_set)}")
        log.info(f"Valid set size: {len(self.valid_set)}")

    def get_dataloader(self, dataset: Dataset, flag: str) -> DataLoader:
        return DataLoader(
            dataset,
            shuffle=flag == "train",
            drop_last=flag == "train",
            **self.loader_kwargs,
            collate_fn=partial(collate_and_transform, transforms=self.transforms),
        )

    def train_dataloader(self) -> DataLoader:
        return self.get_dataloader(self.train_set, "train")

    def val_dataloader(self) -> DataLoader:
        return self.get_dataloader(self.valid_set, "valid")

    def test_dataloader(self) -> DataLoader:
        return self.val_dataloader()

    def predict_dataloader(self) -> DataLoader:
        return self.test_dataloader()

    def get_sample(self) -> dict:
        return next(iter(self.train_set))


class DijetModule(LightningDataModule):
    """Datamodule for the processed dijet data."""

    def __init__(
        self,
        *,
        data_dir: str,
        sig_file: str,
        bkg_file: str,
        extra_test: str,
        loader_kwargs: dict,
        mjj_window: tuple | list | None = None,
        n_sig: int | None = None,
        n_bkg: int | None = None,
        n_dope: int | None = None,
        n_csts: int | None = 0,
        num_folds: int = 5,
        test_fold: int = 0,
    ) -> None:
        super().__init__()
        self.loader_kwargs = loader_kwargs

        # Load the full combined dataset
        dataset = load_strong_cwola_data(
            Path(data_dir, bkg_file),
            Path(data_dir, sig_file),
            mjj_window,
            n_sig,
            n_bkg,
            n_dope,
            n_csts,
        )

        # Calculate the positive weight to balance the classes
        n_bkg = (dataset["cwola_labels"] == 0).sum()
        n_sig = (dataset["cwola_labels"] == 1).sum()
        self.pos_weight = n_bkg / n_sig  # This is called in the model on_fit_start

        # Split the dataset into train, valid and test based on the test fold intex
        train_set, valid_set, test_set = k_fold_split(dataset, num_folds, test_fold)
        self.train_set = DictDataset(train_set)
        self.valid_set = DictDataset(valid_set)
        self.test_set = DictDataset(test_set)
        log.info(f"Train set size: {len(self.train_set)}")
        log.info(f"Valid set size: {len(self.valid_set)}")
        log.info(f"Test set size: {len(self.test_set)}")

        # Load the extra test data
        extra_test = load_dijet_file(
            Path(data_dir, extra_test),
            mjj_window,
            None,  # All events
            n_csts,
        )
        extra_test["cwola_labels"] = np.zeros_like(extra_test["labels"])
        self.extra_test = DictDataset(extra_test)
        log.info(f"Extra test set size: {len(self.extra_test)}")

    def get_dataloader(self, dataset: Dataset, flag: str) -> DataLoader:
        return DataLoader(
            dataset,
            shuffle=flag == "train",
            drop_last=flag == "train",
            **self.loader_kwargs,
        )

    def train_dataloader(self) -> DataLoader:
        return self.get_dataloader(self.train_set, "train")

    def val_dataloader(self) -> DataLoader:
        return self.get_dataloader(self.valid_set, "valid")

    def test_dataloader(self) -> DataLoader:
        return [
            self.get_dataloader(self.test_set, "test"),
            self.get_dataloader(self.extra_test, "extra_test"),
        ]

    def predict_dataloader(self) -> DataLoader:
        return self.test_dataloader()

    def get_sample(self) -> dict:
        return next(iter(self.train_set))
