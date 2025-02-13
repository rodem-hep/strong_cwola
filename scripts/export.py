import logging

import hydra
import rootutils
from omegaconf import DictConfig

root = rootutils.setup_root(search_from=__file__, pythonpath=True)

from mltools.mltools.hydra_utils import reload_original_config
from mltools.mltools.lightning_utils import save_predictions

log = logging.getLogger(__name__)
cfg_path = str(root / "configs")


@hydra.main(version_base=None, config_path=cfg_path, config_name="export.yaml")
def main(cfg: DictConfig) -> None:
    log.info("Loading run information")
    orig_cfg = reload_original_config(ckpt_flag=cfg.ckpt_flag)

    log.info("Loading best checkpoint")
    model_class = hydra.utils.get_class(orig_cfg.model._target_)
    model = model_class.load_from_checkpoint(orig_cfg.ckpt_path, map_location="cpu")

    log.info("Instantiating original trainer")
    trainer = hydra.utils.instantiate(orig_cfg.trainer)

    log.info("Instantiating the original datamodule")
    datamodule = hydra.utils.instantiate(orig_cfg.datamodule)

    log.info("Saving predictions")
    save_predictions(model, datamodule, trainer, orig_cfg.full_path)


if __name__ == "__main__":
    main()
