from functools import partial

import torch as T
import torch.nn.functional as F
from lightning import LightningModule

from mltools.mltools.mlp import MLP
from mltools.mltools.modules import IterativeNormLayer
from mltools.mltools.transformers import ClassAttentionPooling, Transformer


class Classifier(LightningModule):
    """Binary discriminator for the Low level cwola data."""

    def __init__(
        self,
        *,
        data_sample: dict,
        embed_config: dict,
        encoder_config: dict,
        ca_config: dict,
        optimizer: partial,
        scheduler: partial,
    ) -> None:
        super().__init__()
        self.save_hyperparameters(logger=False)
        self.optimizer = optimizer
        self.scheduler = scheduler

        # Decode the data sample to get the dimensions
        self.cst_dim = data_sample["jet1_locals"].shape[-1]
        self.hlv_dim = data_sample["hlv1"].shape[-1]
        self.mjj_dim = data_sample["mjj"].shape[-1]
        self.ctxt_dim = self.hlv_dim * 2 + self.mjj_dim  # Combined context

        # Normalisation layers
        self.cst_norm = IterativeNormLayer(self.cst_dim)
        self.ctxt_norm = IterativeNormLayer(self.ctxt_dim)

        # The single transformer encoder for the constituents
        self.encoder = Transformer(self.cst_dim, **encoder_config)

        # The embedders for the constituents and the hlv
        self.cst_embed = MLP(self.cst_dim, self.encoder.inpt_dim, **embed_config)
        self.ctxt_embed = MLP(self.ctxt_dim, self.encoder.ctxt_dim, **embed_config)

        # The class attention layer
        self.ca = ClassAttentionPooling(
            inpt_dim=self.encoder.outp_dim,
            outp_dim=1,
            **ca_config,
        )

    def forward(self, data: dict) -> T.Tensor:
        """Pass through the network."""
        csts1 = data["jet1_locals"]
        csts2 = data["jet2_locals"]
        hlv1 = data["hlv1"]
        hlv2 = data["hlv2"]
        mjj = data["mjj"]

        # The masks for the constituents
        mask1 = csts1[..., 0] >

        # Concatenate the two jets - one monolithic input for one monolithic model
        csts = T.cat([csts1, csts2], dim=1)  # Batch x N x D
        hlv = T.cat([hlv1, hlv2, mjj], dim=1)  # Batch x D
        csts = self.cst_norm(csts)  # Normalise
        hlv = self.ctxt_norm(hlv)
        csts = self.cst_embed(csts)  # Embed
        hlv = self.ctxt_embed(hlv)
        x = self.encoder.forward(csts, ctxt=hlv)  # Main transformer
        return self.ca(x)  # Class attention

    def _shared_step(self, data: dict, flag: str) -> T.Tensor:
        outputs = self.forward(data)
        targets = data["cwola_label"]
        loss = F.binary_cross_entropy(outputs, targets)
        self.log(f"{flag}/loss", loss)
        return loss

    def training_step(self, data: dict) -> T.Tensor:
        return self._shared_step(data, "train")

    def validation_step(self, data: dict) -> T.Tensor:
        return self._shared_step(data, "val")

    def predict_step(self, data: dict) -> None:
        """Get the outputs and return all variables needed for saving."""
        outputs = self.forward(data)
        targets = data["cwola_label"]
        labels = data["labels"]
        return {"outputs": outputs, "targets": targets, "labels": labels}

    def configure_optimizers(self) -> dict:
        params = filter(lambda p: p.requires_grad, self.parameters())
        opt = self.optimizer(params)
        sched = self.scheduler(optimizer=opt, model=self)
        return [opt], [{"scheduler": sched, "interval": "step"}]
