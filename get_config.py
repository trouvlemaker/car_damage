import os
from glob import glob

import mmcv
import torch

from utility.custom_utils import setup_logger

logger = setup_logger(name=__name__)


def get_config(soda_cfg, is_train=True):
    load_from = None
    resume_from = None
    if soda_cfg.sodaflow.start_ckpt_path != "" and is_train:
        model_path = glob(os.path.join(soda_cfg.sodaflow.start_ckpt_path, "*.pth"))[0]
        if soda_cfg.train_inputs.FT:
            load_from = model_path
        else:
            resume_from = model_path
    if soda_cfg.sodaflow.test_model_path != "" and not is_train:
        model_path = glob(os.path.join(soda_cfg.sodaflow.test_model_path, "best*.pth"))[
            0
        ]

    # setting default config
    if is_train:
        basic_config = mmcv.Config.fromfile(soda_cfg.model_config)
    else:
        basic_config = mmcv.Config.fromfile(soda_cfg.eval_inputs.model_config)
        basic_config.fuse_conv_bn = soda_cfg.eval_inputs.fuse_conv_bn
    basic_config.checkpoint = model_path
    basic_config.load_from = load_from
    basic_config.resume_from = resume_from
    basic_config.gpu_ids = range(0, torch.cuda.device_count())
    basic_config.device = "cuda" if torch.cuda.is_available() else "cpu"
    basic_config.log_level = soda_cfg.log_level

    # update config
    if not is_train:
        basic_config.show = soda_cfg.eval_inputs.show
        basic_config.show_dir = soda_cfg.eval_inputs.show_dir
        basic_config.show_score_thr = soda_cfg.eval_inputs.show_score_thr
        basic_config.tmpdir = soda_cfg.eval_inputs.tmpdir
        basic_config.gpu_collect = soda_cfg.eval_inputs.gpu_collect
        basic_config.eval_options = soda_cfg.eval_inputs.eval_options
        basic_config.format_only = soda_cfg.eval_inputs.format_only
        basic_config.result_out = soda_cfg.eval_inputs.result_out
        basic_config.eval = soda_cfg.eval_inputs.eval

    # dataset config
    data_root = soda_cfg.sodaflow.dataset_path[0]
    classes = list(soda_cfg.train_inputs.classes)
    assert (
        classes is not None or soda_cfg.train_inputs.num_classes
    ), "classes or num_classes must be not None"
    if soda_cfg.train_inputs.classes is not None:
        num_classes = len(soda_cfg.train_inputs.classes)
    else:
        num_classes = soda_cfg.train_inputs.num_classes

    train_ann = os.path.join(data_root, soda_cfg.train_inputs.train_ann)
    train_imgs = os.path.join(data_root, soda_cfg.train_inputs.train_imgs)
    valid_ann = os.path.join(data_root, soda_cfg.train_inputs.valid_ann)
    valid_imgs = os.path.join(data_root, soda_cfg.train_inputs.valid_imgs)
    if is_train:
        test_ann = os.path.join(data_root, soda_cfg.train_inputs.test_ann)
        test_imgs = os.path.join(data_root, soda_cfg.train_inputs.test_imgs)
        basic_config.data.samples_per_gpu = soda_cfg.train_inputs.batch_size
        basic_config.data.workers_per_gpu = soda_cfg.train_inputs.workers
    else:
        test_ann = os.path.join(data_root, soda_cfg.eval_inputs.test_ann)
        test_imgs = os.path.join(data_root, soda_cfg.eval_inputs.test_imgs)
        basic_config.data.samples_per_gpu = soda_cfg.eval_inputs.batch_size
        basic_config.data.workers_per_gpu = soda_cfg.eval_inputs.workers

    basic_config.data.train.type = soda_cfg.train_inputs.dataset_type
    basic_config.data.train.ann_file = train_ann
    basic_config.data.train.img_prefix = train_imgs
    basic_config.data.train.classes = classes

    basic_config.data.val.type = soda_cfg.train_inputs.dataset_type
    basic_config.data.val.ann_file = valid_ann
    basic_config.data.val.img_prefix = valid_imgs
    basic_config.data.val.classes = classes
    if is_train:
        basic_config.data.test.type = soda_cfg.train_inputs.dataset_type
    else:
        basic_config.data.test.type = soda_cfg.eval_inputs.dataset_type
    basic_config.data.test.ann_file = test_ann
    basic_config.data.test.img_prefix = test_imgs
    basic_config.data.test.classes = classes

    if is_train:
        basic_config.launcher = soda_cfg.train_inputs.lanucher
        if len(basic_config.gpu_ids) > 1:
            basic_config.launcher = "pytorch"
    else:
        basic_config.launcher = soda_cfg.eval_inputs.lanucher
    basic_config.deterministic = False
    basic_config.exp_name = soda_cfg.experiment_name

    # model_config
    basic_config.model.bbox_head.num_classes = num_classes
    if is_train:
        basic_config.model.test_cfg.score_thr = soda_cfg.train_inputs.test_score_thr
    else:
        basic_config.model.test_cfg.score_thr = soda_cfg.eval_inputs.test_score_thr

    # running config
    max_epoch = soda_cfg.train_inputs.max_epoch
    # TODO: Add mapping optim and lr from yaml to mmdet config python file
    # CosineRestart
    basic_config.lr_config.warmup_ratio = soda_cfg.train_inputs.warmup_ratio
    basic_config.lr_config.warmup_iters = soda_cfg.train_inputs.warmup_iters
    basic_config.lr_config.periods = list(soda_cfg.train_inputs.lr_periods)
    basic_config.lr_config.restart_weights = list(
        soda_cfg.train_inputs.lr_restart_weights
    )
    basic_config.lr_config.min_lr_ratio = soda_cfg.train_inputs.min_lr_ratio
    # basic_config.lr_config.step = list(soda_cfg.train_inputs.lr_step)
    # basic_config.lr_config.num_last_epochs = soda_cfg.train_inputs.num_last_epochs
    basic_config.runner.max_epochs = max_epoch
    basic_config.optimizer.lr = float(soda_cfg.train_inputs.lr)
    basic_config.log_config.interval = int(soda_cfg.train_inputs.log_interval)
    basic_config.checkpoint_config.interval = int(soda_cfg.train_inputs.ckpt_interval)
    basic_config.checkpoint_config.max_keep_ckpts = soda_cfg.train_inputs.max_keep_ckpts
    basic_config.evaluation.interval = soda_cfg.train_inputs.eval_interval
    if soda_cfg.train_inputs.save_best_metric:
        basic_config.evaluation.save_best = soda_cfg.train_inputs.save_best_metric
        basic_config.custom_hooks[-1].save_best = True
    for cst_hook in basic_config.custom_hooks:
        if cst_hook.type == "SodaflowTrackingHook":
            cst_hook.interval = basic_config.log_config.interval
    basic_config.work_dir = soda_cfg.work_dir

    logger.info("================== Configs ==================")
    logger.info(basic_config)

    return basic_config
