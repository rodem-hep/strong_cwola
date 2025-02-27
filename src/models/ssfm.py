from functools import partial

import torch as T
from lightning import LightningModule
from torch import nn

from mltools.mltools.mlp import MLP
from mltools.mltools.modules import Fourier, IterativeNormLayer
from mltools.mltools.torch_utils import append_dims
from mltools.mltools.transformers import Transformer
from src.models.utils import JetBackbone


class SSFM(LightningModule):
    """Class for the set-to-set flow modelling pre-training."""

    def __init__(
        self,
        *,
        data_sample: dict,
        embed_config: dict,
        encoder_config: dict,
        decoder_config: dict,
        optimizer: partial,
        scheduler: partial,
    ) -> None:
        super().__init__()
        self.save_hyperparameters(logger=False)
        self.optimizer = optimizer
        self.scheduler = scheduler

        # Decode the data sample to get the dimensions
        self.csts_dim = data_sample["csts"].shape[-1]
        self.ctxt_dim = data_sample["ctxt"].shape[-1]

        # The transformers
        self.encoder = Transformer(**encoder_config)
        self.decoder = Transformer(
            inpt_dim=self.csts_dim,
            outp_dim=self.csts_dim * 2,  # Includes the logvar
            use_decoder=True,
            **decoder_config,
        )

        # The embedding and normalisation layers
        self.csts_norm = IterativeNormLayer(self.csts_dim)
        self.ctxt_norm = IterativeNormLayer(self.ctxt_dim)
        self.csts_embed = MLP(self.csts_dim, self.encoder.inpt_dim, **embed_config)
        self.ctxt_embed = MLP(self.ctxt_dim, self.encoder.ctxt_dim, **embed_config)

        # The decoder needs an additional embedding for the time
        self.time_emb = nn.Sequential(
            Fourier(16), MLP(16, self.decoder.ctxt_dim, **embed_config)
        )

        # Keep everything packed, this saves all the memory!!!
        # Does require and ampere GPU though...
        self.encoder.pack_inputs = True
        self.decoder.pack_inputs = True
        self.encoder.unpack_output = False
        self.decoder.unpack_output = False

    def forward(self, batch: dict) -> T.Tensor:
        """Pass through the network."""
        csts = batch["csts"]
        mask = batch["mask"]
        ctxt = batch["ctxt"]
        null_mask = batch["null_mask"]

        # Training needs the output packed - this could have changed during save
        self.encoder.unpack_output = False

        # Split the jets into the two sets (done by masking only)
        enc_mask = mask & ~null_mask
        dec_mask = mask & null_mask

        # Get all the values required for the diffusion / flow matching
        x0 = csts  # Clean sample
        x1 = T.randn_like(x0)  # Random sample
        t = T.sigmoid(T.randn(x0.shape[0], device=x0.device))  # Sample time
        t_emb = self.time_emb(t)  # Time embedding
        t = append_dims(t, x0.ndim)  # Match dimensions for interpolation
        xt = (1 - t) * x0 + t * x1  # Interpolate
        v = x1[dec_mask] - x0[dec_mask]  # Velocity vector for target

        # Pass through the encoder
        csts = self.csts_norm(csts, mask)  # Normalise
        ctxt = self.ctxt_norm(ctxt)
        csts = self.csts_embed(csts)  # Embed
        ctxt = self.ctxt_embed(ctxt)
        enc_out, enc_culens, enc_maxlen = self.encoder(csts, ctxt=ctxt, mask=enc_mask)

        # Get the output of the decoder using, time and context
        dec_out, _, _ = self.decoder(
            xt,
            mask=dec_mask,
            ctxt=t_emb,
            kv=enc_out,
            kv_culens=enc_culens,
            kv_maxlen=enc_maxlen,
        )

        # Calculate the loss based on the velocity vector
        v_hat, log_var = dec_out.chunk(2, dim=-1)
        return ((v - v_hat).square() / log_var.exp() + log_var).mean()

    def training_step(self, batch: dict) -> T.Tensor:
        loss = self.forward(batch)
        self.log("train/loss", loss)
        return loss

    def validation_step(self, data: dict, batch_idx: int) -> T.Tensor:
        loss = self.forward(data)
        self.log("valid/loss", loss)
        return loss

    def configure_optimizers(self) -> dict:
        params = filter(lambda p: p.requires_grad, self.parameters())
        opt = self.hparams.optimizer(params)
        sched = self.hparams.scheduler(optimizer=opt, model=self)
        return [opt], [{"scheduler": sched, "interval": "step"}]

    def on_validation_epoch_end(self) -> None:
        """Create the pickled object for the backbone."""
        backbone = JetBackbone(
            self.csts_norm,
            self.ctxt_norm,
            self.csts_embed,
            self.ctxt_embed,
            self.encoder,
        )
        backbone.encoder.unpack_output = True  # Safe than sorry
        backbone.eval()
        T.save(backbone, "backbone.pkl")
