import hydra
import utils
import torch
import logging
from core import train, checkpoint_train
logger = logging.getLogger(__name__)
device = "cuda" if torch.cuda.is_available() else "cpu"
torch.set_float32_matmul_precision('high')

@hydra.main(config_path="cfgs", config_name="config", version_base="1.3")
def main(cfg):
    log_dict = utils.get_log_dict()
    if cfg.checkpoint.ifuse == True:
        checkpoint_train(cfg, cfg.seeds, log_dict, -1, logger, None, hydra.utils.get_original_cwd())
    else:
        train(cfg, cfg.seeds, log_dict, -1, logger, None, hydra.utils.get_original_cwd())


if __name__ == "__main__":
    main()
