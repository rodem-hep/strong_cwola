from functools import partial

import torch as T
from lightning import LightningModule

from mltools.mltools.loss import sigmoid_focal_loss
from mltools.mltools.torch_utils import to_device
from mltools.mltools.transformers import ClassAttentionPooling
from src.models.utils import JetBackbone, calculate_signal_efficiency


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
        backbone_path: str | None = None,
    ) -> None:
        super().__init__()
        self.save_hyperparameters(logger=False)
        self.optimizer = optimizer
        self.scheduler = scheduler
        self.pos_weight = 1  # Placeholder - Set in on_fit_start()

        # Decode the data sample to get the dimensions
        self.csts_dim = data_sample["csts"].shape[-1]
        self.ctxt_dim = data_sample["ctxt"].shape[-1]

        # Initialise the backbone of the model
        if backbone_path is not None:
            self.backbone = JetBackbone.from_path(backbone_path)
        else:
            self.backbone = JetBackbone.from_config(
                self.csts_dim, self.ctxt_dim, embed_config, encoder_config
            )

        # The class attention layer for pooling
        self.ca = ClassAttentionPooling(
            inpt_dim=self.backbone.outp_dim,
            outp_dim=1,
            **ca_config,
        )

        # Metrics for the model
        self.val_outs = []
        self.val_true_labels = []
        self.val_cwola_labels = []

    def on_fit_start(self) -> None:
        """Get the positive weight from the associated datamodule."""
        self.pos_weight = T.tensor(
            self.trainer.datamodule.pos_weight,
            device=self.device,
            dtype=T.float32,
        )

    def forward(self, batch: dict) -> T.Tensor:
        """Pass through the network."""
        x, m = self.backbone(batch["csts"], batch["mask"], batch["ctxt"])
        return self.ca(x, mask=m)  # Class attention

    def _shared_step(self, batch: dict, flag: str) -> T.Tensor:
        outputs = self.forward(batch)
        targets = batch["cwola_labels"]
        loss = sigmoid_focal_loss(
            outputs.squeeze(), targets, pos_weight=self.pos_weight
        )
        self.log(f"{flag}/loss", loss)

        if flag == "valid":
            self.val_outs.append(outputs.squeeze())
            self.val_true_labels.append(batch["labels"])
            self.val_cwola_labels.append(batch["cwola_labels"])

        return loss

    def training_step(self, batch: dict) -> T.Tensor:
        return self._shared_step(batch, "train")

    def validation_step(self, batch: dict) -> T.Tensor:
        return self._shared_step(batch, "valid")

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

    def predict_step(
        self,
        batch: dict,
        batch_idx: int = 0,
        dataloader_idx: int = 0,
    ) -> dict:
        """Get the outputs and return all variables needed for saving."""
        outputs = self.forward(batch)
        outdict = {
            "outputs": outputs,
            "target": batch["cwola_labels"].view(-1, 1),
            "weight": T.ones_like(outputs),
            "labels": batch["labels"].view(-1, 1),
            "event_ids": batch["event_ids"],
            "mjj": batch["mjj"],
            "is_pythia": batch["is_pythia"].view(-1, 1),
        }
        return to_device(outdict, "cpu")  # Convert now or we risk running out of VRAM!

    def configure_optimizers(self) -> dict:
        params = filter(lambda p: p.requires_grad, self.parameters())
        opt = self.optimizer(params)
        sched = self.scheduler(optimizer=opt, model=self)
        return [opt], [{"scheduler": sched, "interval": "step"}]
