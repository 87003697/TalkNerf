import hydra
from hydra.core.config_store import ConfigStore
from hydra.utils import instantiate
import numpy as np
import torch
import logging
from omegaconf import DictConfig
import importlib

from engine.basic_engine import * 
log = logging.getLogger(__name__)


@hydra.main(config_path='configs', config_name=None)
def main(cfg: DictConfig):
    log.info(OmegaConf.to_yaml(cfg))
                  
    # Set the relevant seeds for reproducibility
    np.random.seed(cfg.seed)
    torch.manual_seed(cfg.seed)
    
    # Intialize the engine 
    log.info('Engine specified to be {}'.format(cfg.engine))
    _engine_module = importlib.import_module('engine.{}'.format(cfg.engine))
    engine = _engine_module.NerfEngine(cfg = cfg)
    
    engine.load_dataset()
    engine.build_networks()
    engine.setup_optimizer()
    engine.restore_checkpoint()
    engine.setup_scheduler()
    engine.setup_visualizer()
    engine.train()
    engine.evaluate_full()
    engine.videos_synthesis()

if __name__ == '__main__':
    main()