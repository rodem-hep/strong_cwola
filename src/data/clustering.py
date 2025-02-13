import math

import awkward as ak
import fastjet
import numpy as np
from vector import MomentumObject4D
from vector.backends.awkward import MomentumArray4D


def signed_angle_diff(angle1: float, angle2: float = 0) -> float:
    """Calculate diff between two angles reduced to the interval of [-pi, pi]."""
    return (angle1 - angle2 + math.pi) % (2 * math.pi) - math.pi


def convert_to_relative(csts: np.ndarray, jets: np.ndarray) -> None:
    """Inplace conversion of the jet tensor to the relative coordinates."""
    csts[..., 0] /= jets[..., 0][:, None]
    csts[..., 1] -= jets[..., 1][:, None]
    csts[..., 2] = signed_angle_diff(csts[..., 2], jets[..., 2][:, None])
    csts[:] *= csts[..., 0:1] > 0  # Force zero padding


def make_ragged(x: ak.Array, mask: np.ndarray, axis: int = 1) -> ak.Array:
    """Make a ragged array."""
    x = ak.mask(x, mask)
    x = ak.from_regular(x, axis=axis)
    return ak.drop_none(x, axis=axis)


def make_padded(x: ak.Array, pad_size: int) -> np.ndarray:
    """Convert to a padded numpy array."""
    x = ak.pad_none(x, target=pad_size, axis=1, clip=True)
    x = ak.to_numpy(x).data
    x = np.array(x)
    x = x.view(np.float32).reshape((*x.shape, -1))
    return np.nan_to_num(x, 0.0)


def cluster_numpy_batch(
    csts: np.ndarray,
    jetdef: fastjet.JetDefinition,
    mask: np.ndarray | None = None,
) -> fastjet.ClusterSequence:
    """Run fastjet clustering on the numpy array of constituents.

    Meant to run on a batch of events where csts has shape B x N x D.
    The first three dimensions are taken to be the pt, eta, phi of the constituents.
    All constituents are assumed to be massless.
    """
    ak_array = ak.from_numpy(csts)  # FastJet requires an awkward array

    # Mask the constituents if necessary - this creates a ragged awkward array
    if mask is not None:
        ak_array = make_ragged(ak_array, mask)

    # Zip them together for FastJet
    ak_array = ak.zip(
        {
            "pt": ak_array[:, :, 0],
            "eta": ak_array[:, :, 1],
            "phi": ak_array[:, :, 2],
            "mass": ak.zeros_like(ak_array[:, :, 0]),  # Massless
        },
        with_name="Momentum4D",
    )

    # Run the clustering
    return fastjet.ClusterSequence(ak_array, jetdef)


def calc_delta_r(x: MomentumObject4D, y: MomentumObject4D) -> np.ndarray:
    """Calculate delta R."""
    return np.sqrt((x.eta - y.eta) ** 2 + (x.phi - y.phi) ** 2)


def n_subjettiness(
    axes: MomentumArray4D,
    csts: MomentumArray4D,
    R: float = 1.0,
) -> float:
    """Calculate the n-subjettiness of the constituents with respect to the axes.

    axes should have shape Events * Njets * NA * Momentum4D
    csts should have shape Events * Njets * NC * Momentum4D
    """
    delta_r = calc_delta_r(axes[..., None], csts[..., None, :])
    delta_r = ak.min(delta_r, axis=2)
    d0 = ak.sum(csts.pt, -1) * R
    return ak.sum(csts.pt * delta_r, -1) / d0
