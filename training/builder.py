import datetime
from dateutil import tz
from omegaconf import OmegaConf
import wandb

from datasets import ClipCocoDataset

def load_cfg(args):
    cfg = OmegaConf.load(args.cfg_path)
    
    cfg.data.sample_frac = args.sample_frac
    cfg.data.remove_modality_gap = args.remove_modality_gap
    cfg.data.remove_mean = args.remove_mean
    cfg.data.normalize_prefix = args.normalize_prefix
    cfg.data.add_gaussian_noise = args.add_gaussian_noise
    cfg.data.test_split = args.test_split
    
    # get current time
    now = datetime.datetime.now(tz.tzlocal())
    timestamp = now.strftime("%Y_%m_%d_%H_%M_%S")
    
    if cfg.data.normalize_prefix:
        cfg.experiment_name += "_normed"
        
    if not OmegaConf.is_none(cfg.data, "sample_frac"):
        cfg.experiment_name += f"_frac{cfg.data.sample_frac}"

    if cfg.data.remove_modality_gap:
        cfg.experiment_name += f"_remove_modality_gap"
        
    if cfg.data.remove_mean:
        cfg.experiment_name += f"_remove_mean"
        
    if cfg.data.add_gaussian_noise:
        cfg.experiment_name += f"_add_gaussian_noise"
    
    cfg.experiment_name = f"{cfg.experiment_name}/{timestamp}"
    
    return cfg
    

def build_dataset(cfg, split):
    return ClipCocoDataset(cfg, split)

def build_wandb_logger(cfg, args):
    # wandb logging
    wandb.init(
        name=cfg.experiment_name,
        project=cfg.logger.project,
        entity=cfg.logger.entity,
        dir=cfg.logger.save_dir
    )
    wandb.config.update(args)