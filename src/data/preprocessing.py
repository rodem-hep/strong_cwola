from collections.abc import Iterable

import numpy as np
import torch as T
from torch.utils.data import default_collate


def collate_and_transform(
    batch: Iterable[dict],
    do_default_collate: bool = True,
    transforms: list[callable] | None = None,
) -> dict:
    """Collate the batch and apply the transforms.

    Why this not lightning's on_before_batch_transfer?
     - This still runs inside the pytorch multiprocessing pool for data loading.
     - Thus it runs asynchonously for each batch being prepared.
    """
    if do_default_collate:
        batch = default_collate(batch)
    if transforms is not None:
        for transform in transforms:
            batch = transform(batch)
    return batch


def mask_batch(
    jet_dict: dict,
    mask_fraction: float | None = 0.5,
    key: str = "null_mask",
) -> dict:
    """Applies a masking function of a batch of jets.

    Will add a new key to the jet_dict with the locations of the new mask.
    """
    null_mask = T.stack([
        mask_jet(mask, mask_fraction=mask_fraction) for mask in jet_dict["mask"]
    ])
    jet_dict[key] = null_mask
    return jet_dict


def mask_jet(
    mask: T.Tensor,
    mask_fraction: float | None,
    seed: int | None = None,
) -> np.ndarray:
    """Randomly drop a fraction of the jet based on the total number of constituents."""
    if seed is not None:
        T.manual_seed(seed)
    if mask_fraction is None:
        mask_fraction = T.rand(1).item()

    # Calculate the number of nodes to drop
    n_drop = int(mask.sum() * mask_fraction)
    n_drop = max(1, min(n_drop, mask.sum() - 1))  # At least one drop / survive

    # Generate a random score per node, the lowest frac will be killed
    score = T.rand(len(mask))
    score[~mask] = 9999
    drop_idx = T.argsort(score)[:n_drop]

    # Create the null mask: True = dropped
    null_mask = T.zeros_like(mask, dtype=T.bool)
    null_mask[drop_idx] = True
    return null_mask
