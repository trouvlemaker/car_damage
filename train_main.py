import copy
import os.path as osp
import time

import mmcv
import sodaflow
from mmcv.runner import get_dist_info, init_dist
from mmcv.utils import get_git_hash
from mmdet import __version__
from mmdet.apis import init_random_seed, set_random_seed, train_detector
from mmdet.datasets import build_dataset
from mmdet.models import build_detector
from mmdet.utils import collect_env, get_root_logger
from omegaconf import DictConfig

from get_config import get_config


def train(config):
    # init distributed env first, since logger depends on the dist info.
    if config.launcher == "none":
        distributed = False
    else:
        distributed = True
        init_dist(config.launcher, **config.dist_params)
        # re-set gpu_ids with distributed training mode
        _, world_size = get_dist_info()
        config.gpu_ids = range(world_size)

    # create work_dir
    mmcv.mkdir_or_exist(osp.abspath(config.work_dir))
    # dump config
    config.dump(osp.join(config.work_dir, "train_config.py"))
    # init the logger before other steps
    timestamp = time.strftime("%Y%m%d_%H%M%S", time.localtime())
    log_file = osp.join(config.work_dir, f"{timestamp}.log")
    logger = get_root_logger(log_file=log_file, log_level=config.log_level)

    # init the meta dict to record some important information such as
    # environment info and seed, which will be logged
    meta = dict()
    # log env info
    env_info_dict = collect_env()
    env_info = "\n".join([(f"{k}: {v}") for k, v in env_info_dict.items()])
    dash_line = "-" * 60 + "\n"
    logger.info("Environment info:\n" + dash_line + env_info + "\n" + dash_line)
    meta["env_info"] = env_info
    meta["config"] = config.pretty_text
    # log some basic info
    logger.info(f"Distributed training: {distributed}")
    logger.info(f"Config:\n{config.pretty_text}")

    # set random seeds
    seed = init_random_seed(None)
    logger.info(f"Set random seed to {seed}, " f"deterministic: {config.deterministic}")
    set_random_seed(seed, deterministic=config.deterministic)
    config.seed = seed
    meta["seed"] = seed
    meta["exp_name"] = osp.basename(config.exp_name)

    model = build_detector(
        config.model, train_cfg=config.get("train_cfg"), test_cfg=config.get("test_cfg")
    )
    model.init_weights()

    datasets = [build_dataset(config.data.train)]
    if len(config.workflow) == 2:
        val_dataset = copy.deepcopy(config.data.val)
        val_dataset.pipeline = config.data.train.pipeline
        datasets.append(build_dataset(val_dataset))
    if config.checkpoint_config is not None:
        # save mmdet version, config file content and class names in
        # checkpoints as meta data
        config.checkpoint_config.meta = dict(
            mmdet_version=__version__ + get_git_hash()[:7], CLASSES=datasets[0].CLASSES
        )
    # add an attribute for visualization convenience
    model.CLASSES = datasets[0].CLASSES
    train_detector(
        model,
        datasets,
        config,
        distributed=distributed,
        validate=True,
        timestamp=timestamp,
        meta=meta,
    )


@sodaflow.main(config_path="./configs", config_name="train_config")
def run_app(cfg: DictConfig) -> None:
    mmcfg = get_config(cfg)
    train(mmcfg)
    print("All Training Done.")


if __name__ == "__main__":
    run_app()
