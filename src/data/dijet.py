import logging
from copy import deepcopy
from pathlib import Path

from lightning import LightningDataModule
from torch.utils.data import DataLoader, Dataset

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


class DijetModule(LightningDataModule):
    """Datamodule for the processed dijet data."""

    def __init__(
        self,
        *,
        data_dir: str,
        sig_file: str,
        bkg_file: str,
        extra_test_file: str,
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
        self.extra_test = load_dijet_file(
            Path(data_dir, extra_test_file),
            mjj_window,
            None,  # All events
            n_csts,
        )
        log.info(f"Extra test set size: {len(self.extra_test)}")

    def train_dataloader(self) -> DataLoader:
        return DataLoader(self.train_set, **self.loader_kwargs, shuffle=True)

    def val_dataloader(self) -> DataLoader:
        val_kwargs = deepcopy(self.loader_kwargs)
        val_kwargs["drop_last"] = False
        val_kwargs["shuffle"] = False
        return DataLoader(self.valid_set, **val_kwargs)

    def test_dataloader(self) -> DataLoader:
        test_kwargs = deepcopy(self.loader_kwargs)
        test_kwargs["drop_last"] = False
        test_kwargs["shuffle"] = False
        return [
            DataLoader(self.test_set, **test_kwargs),
            DataLoader(self.extra_test, **test_kwargs),
        ]

    def predict_dataloader(self) -> DataLoader:
        return self.test_dataloader()

    def get_sample(self) -> dict:
        return next(iter(self.train_set))
