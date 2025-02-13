"""Basic training script."""

import logging

import hydra
import lightning.pytorch as pl
import rootutils
import torch as T
from omegaconf import DictConfig

root = rootutils.setup_root(search_from=__file__, pythonpath=True)

from mltools.mltools.hydra_utils import (
    instantiate_collection,
    log_hyperparameters,
    print_config,
    reload_original_config,
    save_config,
)
from mltools.mltools.lightning_utils import save_predictions
from mltools.mltools.utils import save_declaration

log = logging.getLogger(__name__)
cfg_path = str(root / "configs")


@hydra.main(version_base=None, config_path=cfg_path, config_name="train.yaml")
def main(cfg: DictConfig) -> None:
    """Main training script."""
    log.info("Setting up full job config")

    if cfg.full_resume:
        log.info("Attempting to resume previous job")
        old_cfg = reload_original_config(ckpt_flag=cfg.ckpt_flag)
        if old_cfg is not None:
            cfg = old_cfg
    print_config(cfg)

    log.info(f"Setting seed to: {cfg.seed}")
    pl.seed_everything(cfg.seed, workers=True)

    log.info(f"Setting matrix precision to: {cfg.precision}")
    T.set_float32_matmul_precision(cfg.precision)

    log.info("Instantiating the data module")
    datamodule = hydra.utils.instantiate(cfg.datamodule)

    if cfg.ckpt_path is not None:
        log.info(f"Loading model from checkpoint: {cfg.ckpt_path}")
        if cfg.ckpt_weights_only:
            log.info("Loading only the weights from the checkpoint")
            model_class = hydra.utils.get_class(cfg.model._target_)
            model = model_class.load_from_checkpoint(cfg.ckpt_path, map_location="cpu")
            ckpt_path = None
        else:
            log.info("Setting training to resume from checkpoint")
            ckpt_path = cfg.ckpt_path
    else:
        log.info("Instantiating new model")
        ckpt_path = cfg.ckpt_path
        model = hydra.utils.instantiate(cfg.model, data_sample=datamodule.get_sample())

    if cfg.compile:
        log.info(f"Compiling the model using torch 2.0: {cfg.compile}")
        model = T.compile(model, mode=cfg.compile)

    log.info("Instantiating all callbacks")
    callbacks = instantiate_collection(cfg.callbacks)

    log.info("Instantiating the logger")
    logger = hydra.utils.instantiate(cfg.logger)

    log.info("Instantiating the trainer")
    trainer = hydra.utils.instantiate(cfg.trainer, callbacks=callbacks, logger=logger)

    log.info("Logging all hyperparameters")
    log_hyperparameters(cfg, model, trainer)
    log.info(model)

    log.info("Saving config so job can be resumed")
    save_config(cfg)

    log.info("Starting training!")
    trainer.fit(model, datamodule=datamodule, ckpt_path=ckpt_path)

    log.info("Checking if training finished correctly")
    if trainer.state.status == "finished":
        log.info(" -- YES!! -- ")
        save_declaration()

    if cfg.save_test_preds:
        log.info("Running inference on test set")
        log.info("Attempting to load best checkpoint")
        ckpt_path = trainer.checkpoint_callback.best_model_path
        if not ckpt_path:
            log.warning("Best ckpt not found! Using current weights for testing...")
            ckpt_path = None
        save_predictions(model, datamodule, trainer, cfg.full_path, ckpt_path)


if __name__ == "__main__":
    main()
