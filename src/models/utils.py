from typing import Self

import torch as T
from torch import nn

from mltools.mltools.mlp import MLP
from mltools.mltools.modules import IterativeNormLayer
from mltools.mltools.transformers import Transformer


class JetBackbone(nn.Module):
    """Generalised backbone for the jet models.

    Simply wraps the constituent embedding, constituent id embedding and encoder
    together in a single module.
    Easy for saving and loading using the pickle module.
    """

    def __init__(
        self,
        csts_norm: IterativeNormLayer,
        ctxt_norm: IterativeNormLayer,
        csts_embed: MLP,
        ctxt_embed: MLP,
        encoder: Transformer,
    ) -> None:
        super().__init__()
        self.csts_norm = csts_norm
        self.ctxt_norm = ctxt_norm
        self.csts_embed = csts_embed
        self.ctxt_embed = ctxt_embed
        self.encoder = encoder

        # Store the dimensions for convenience
        self.csts_dim = self.csts_embed.inpt_dim
        self.ctxt_dim = self.ctxt_embed.inpt_dim
        self.outp_dim = encoder.outp_dim

    def forward(self, csts: T.Tensor, mask: T.BoolTensor, ctxt: T.Tensor) -> T.Tensor:
        csts = self.csts_norm(csts, mask)  # Normalise
        ctxt = self.ctxt_norm(ctxt)
        csts = self.csts_embed(csts)  # Embed
        ctxt = self.ctxt_embed(ctxt)
        x = self.encoder(csts, ctxt=ctxt, mask=mask)  # Main transformer
        m = self.encoder.get_combined_mask(mask)  # Mask might gain registers
        return x, m

    @classmethod
    def from_config(
        cls,
        csts_dim: int,
        ctxt_dim: int,
        embed_config: dict,
        encoder_config: dict,
    ) -> Self:
        """Create a JetBackbone from configuration parameters."""
        csts_norm = IterativeNormLayer(csts_dim)
        ctxt_norm = IterativeNormLayer(ctxt_dim)
        encoder = Transformer(**encoder_config)
        csts_embed = MLP(csts_dim, encoder.inpt_dim, **embed_config)
        ctxt_embed = MLP(ctxt_dim, encoder.ctxt_dim, **embed_config)
        return cls(csts_norm, ctxt_norm, csts_embed, ctxt_embed, encoder)

    @classmethod
    def from_path(cls, path: str) -> Self:
        """Load a JetBackbone from a file."""
        return T.load(path)


def calculate_signal_efficiency(
    scores: T.Tensor, labels: T.Tensor, rejection_rate: float = 0.999
) -> T.Tensor:
    """Calculate the signal efficiency at a given background rejection rate."""
    # Pull out the background scores and sort them
    bkg_mask = labels == 0
    bkg_scores = T.sort(scores[bkg_mask]).values

    # Find the score that is at the desired rejection rate to get the threshold
    rej_index = int(bkg_scores.shape[0] * rejection_rate)
    threshold = bkg_scores[rej_index]

    # Calculate the signal efficiency at this threshold
    sig_mask = labels == 1
    sig_scores = scores[sig_mask]
    if sig_mask.sum() == 0:  # If the signal is empty, return 0
        return T.tensor(0.0, device=scores.device)
    return (sig_scores > threshold).float().mean()
