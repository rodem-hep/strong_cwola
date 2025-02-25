from functools import partial

import torch as T
import torch.nn.functional as F
from lightning import LightningModule

from mltools.mltools.loss import sigmoid_focal_loss
from mltools.mltools.mlp import MLP
from mltools.mltools.modules import IterativeNormLayer
from mltools.mltools.transformers import ClassAttentionPooling, Transformer
from src.models.utils import calculate_signal_efficiency


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
        self.pos_weight = 1  # Placeholder - Will be set in on_train_start

        # Decode the data sample to get the dimensions
        self.cst_dim = data_sample["csts1"].shape[-1] + 1  # Extra dim for njet flag
        self.hlv_dim = data_sample["jets1"].shape[-1]
        self.mjj_dim = data_sample["mjj"].shape[-1]
        self.ctxt_dim = self.hlv_dim * 2 + self.mjj_dim  # Combined context

        # Normalisation layers
        self.cst_norm = IterativeNormLayer(self.cst_dim)
        self.ctxt_norm = IterativeNormLayer(self.ctxt_dim)

        # The single transformer encoder for the constituents
        self.encoder = Transformer(**encoder_config)

        # The embedders for the constituents and the hlv
        self.cst_embed = MLP(self.cst_dim, self.encoder.inpt_dim, **embed_config)
        self.ctxt_embed = MLP(self.ctxt_dim, self.encoder.ctxt_dim, **embed_config)

        # The class attention layer
        self.ca = ClassAttentionPooling(
            inpt_dim=self.encoder.outp_dim,
            outp_dim=1,
            **ca_config,
        )

        # Metrics for the model
        self.val_outs = []
        self.val_true_labels = []
        self.val_cwola_labels = []

    def on_fit_start(self):
        """Get the positive weight from the associated datamodule."""
        self.pos_weight = T.tensor(
            self.trainer.datamodule.pos_weight,
            device=self.device,
            dtype=T.float32,
        )

    def forward(self, data: dict) -> T.Tensor:
        """Pass through the network."""
        csts1 = data["csts1"]
        csts2 = data["csts2"]
        jets1 = data["jets1"]
        jets2 = data["jets2"]
        mjj = data["mjj"]

        # Concatenate the two jets - one monolithic input for one monolithic model
        csts1 = F.pad(csts1, (0, 1), value=0)  # Pad the last dimension with a njet flag
        csts2 = F.pad(csts2, (0, 1), value=1)
        csts = T.cat([csts1, csts2], dim=1)  # Batch x N x D
        hlv = T.cat([jets1, jets2, mjj], dim=1) * 0  # Batch x D
        mask = csts[..., 0] > 0
        csts = self.cst_norm(csts, mask)  # Normalise
        hlv = self.ctxt_norm(hlv)
        csts = self.cst_embed(csts)  # Embed
        hlv = self.ctxt_embed(hlv)
        x = self.encoder(csts, ctxt=hlv, mask=mask)  # Main transformer
        mask = self.encoder.get_combined_mask(mask)  # Might gain registers
        return self.ca(x, mask=mask)  # Class attention

    def _shared_step(self, data: dict, flag: str) -> T.Tensor:
        outputs = self.forward(data)
        targets = data["cwola_labels"]
        loss = sigmoid_focal_loss(
            outputs.squeeze(),
            targets,
            pos_weight=self.pos_weight,
        )
        self.log(f"{flag}/loss", loss)

        if flag == "valid":
            self.val_outs.append(outputs.squeeze())
            self.val_true_labels.append(data["labels"])
            self.val_cwola_labels.append(data["cwola_labels"])

        return loss

    def training_step(self, data: dict) -> T.Tensor:
        return self._shared_step(data, "train")

    def validation_step(self, data: dict) -> T.Tensor:
        return self._shared_step(data, "valid")

    def on_validation_epoch_end(self):
        # Flatten the lists
        self.val_outs = T.cat(self.val_outs)
        self.val_true_labels = T.cat(self.val_true_labels)
        self.val_cwola_labels = T.cat(self.val_cwola_labels)

        # Calculate the signal efficiency
        true_eff = calculate_signal_efficiency(self.val_outs, self.val_true_labels)
        cwola_eff = calculate_signal_efficiency(self.val_outs, self.val_cwola_labels)
        self.log("valid/true_eff", true_eff)
        self.log("valid/cwola_eff", cwola_eff)

        # Reset the lists
        self.val_outs = []
        self.val_true_labels = []
        self.val_cwola_labels = []

    def predict_step(self, data: dict) -> None:
        """Get the outputs and return all variables needed for saving."""
        outputs = self.forward(data)
        return {
            "outputs": outputs,
            "target": data["cwola_labels"].view(-1, 1),
            "weight": data["weights"],
            "labels": data["labels"].view(-1, 1),
            "event_ids": data["event_ids"],
            "mjj": data["mjj"],
        }

    def configure_optimizers(self) -> dict:
        params = filter(lambda p: p.requires_grad, self.parameters())
        opt = self.optimizer(params)
        sched = self.scheduler(optimizer=opt, model=self)
        return [opt], [{"scheduler": sched, "interval": "step"}]
