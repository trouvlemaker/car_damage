import logging
import os
import shutil
from collections import OrderedDict
from glob import glob

from mmcv.runner import HOOKS, LoggerHook
from mmdet.utils.logger import get_root_logger
from sodaflow import tracking

logger = get_root_logger(log_level=logging.INFO)


@HOOKS.register_module()
class SodaflowTrackingHook(LoggerHook):
    def __init__(
        self,
        by_epoch=True,
        interval=10,
        interval_exp_name=1000,
        file_client_args=None,
        save_best=False,
    ):
        super(SodaflowTrackingHook, self).__init__(interval, by_epoch)
        self.by_epoch = by_epoch
        self.time_sec_tot = 0
        self.interval_exp_name = interval_exp_name
        self.file_client_args = file_client_args
        self.save_best = save_best

    def _get_info(self, runner):
        info_dict = OrderedDict(
            mode=self.get_mode(runner), epoch=self.get_epoch(runner)
        )

        if "eval_iter_num" in runner.log_buffer.output:
            # this doesn't modify runner.iter and is regardless of by_epoch
            cur_iter = runner.log_buffer.output.pop("eval_iter_num")
        else:
            cur_iter = self.get_iter(runner, inner_iter=True)

        ## get log epoch or total_iter
        if self.by_epoch:
            info_dict["iter"] = cur_iter
            info_dict["total_iter"] = len(runner.data_loader) * runner.epoch + cur_iter
        else:
            info_dict["iter"] = cur_iter
            info_dict["total_iter"] = cur_iter

        ## get lr
        # only record lr of the first param group
        cur_lr = runner.current_lr()
        if isinstance(cur_lr, list):
            info_dict["lr"] = cur_lr[0]
        else:
            assert isinstance(cur_lr, dict)
            info_dict["lr"] = {}
            for k, lr_ in cur_lr.items():
                assert isinstance(lr_, list)
                info_dict["lr"].update({k: lr_[0]})

        ## get other outputs
        info_dict = dict(info_dict, **runner.log_buffer.output)

        ## get checkpoint path
        self.out_dir = runner.work_dir

        return info_dict

    def log(self, runner):
        self.infos = self._get_info(runner)

        metric_dict = dict()
        for key, val in self.infos.items():
            if isinstance(val, str):
                continue
            else:
                metric_dict[key] = val

        tracking.log_metrics(step=self.infos["total_iter"], **metric_dict)

    def after_train_epoch(self, runner):
        logger.debug(f"runner meta => {runner.meta['hook_msgs']}")
        runner.log_buffer.average()
        self.log(runner)
        # logger.debug(f"infos => {self.infos}")
        if "last_ckpt" in runner.meta["hook_msgs"].keys():
            last_ckpt = runner.meta["hook_msgs"]["last_ckpt"]
            model_log_root = os.path.dirname(last_ckpt)
            log_path = glob(os.path.join(model_log_root, "*.log*"))[0]
            config_path = glob(os.path.join(model_log_root, "*.py"))[0]
            tracking.log_model(config_path)
            tracking.log_model(log_path)
            save_path = os.path.join(model_log_root, "latest.pth")
            tracking.log_model(save_path)
        # logger.debug(runner.meta['hook_msgs'])
        if self.save_best and "best_ckpt" in runner.meta["hook_msgs"].keys():
            best_ckpt = runner.meta["hook_msgs"]["best_ckpt"]
            model_log_root = os.path.dirname(best_ckpt)
            save_path = os.path.join(model_log_root, "best_ckpt.pth")
            shutil.copy2(best_ckpt, save_path)
            tracking.log_model(save_path)
