# coding: utf-8

import logging
import os
import sys

APP_BASE_PATH = os.path.abspath(os.path.dirname(__file__))
sys.path.append(APP_BASE_PATH)
import os.path as osp
import time
from pathlib import Path

import mmcv
import numpy as np
import pandas as pd
import sodaflow
import torch
from mmcv.cnn import fuse_conv_bn
from mmcv.runner import get_dist_info, init_dist, load_checkpoint, wrap_fp16_model
from mmdet.apis import multi_gpu_test, single_gpu_test
from mmdet.datasets import build_dataloader, build_dataset, replace_ImageToTensor
from mmdet.models import build_detector
from mmdet.utils import (
    build_ddp,
    build_dp,
    compat_cfg,
    get_device,
    setup_multi_processes,
)
from mmdet.utils.logger import get_root_logger
from omegaconf import DictConfig
from sodaflow import tracking

from get_config import get_config

logger = get_root_logger(log_level=logging.INFO)


@sodaflow.main(config_path="./configs", config_name="train_config")
def run_app(cfg: DictConfig) -> None:
    mmcfg = get_config(cfg, is_train=False)
    mmcfg = compat_cfg(mmcfg)
    setup_multi_processes(mmcfg)
    # set cudnn_benchmark
    if mmcfg.get("cudnn_benchmark", False):
        torch.backends.cudnn.benchmark = True

    if "pretrained" in mmcfg.model:
        mmcfg.model.pretrained = None
    elif "init_cfg" in mmcfg.model.backbone:
        mmcfg.model.backbone.init_cfg = None

    if mmcfg.model.get("neck"):
        if isinstance(mmcfg.model.neck, list):
            for neck_cfg in mmcfg.model.neck:
                if neck_cfg.get("rfp_backbone"):
                    if neck_cfg.rfp_backbone.get("pretrained"):
                        neck_cfg.rfp_backbone.pretrained = None
        elif mmcfg.model.neck.get("rfp_backbone"):
            if mmcfg.model.neck.rfp_backbone.get("pretrained"):
                mmcfg.model.neck.rfp_backbone.pretrained = None

    mmcfg.device = get_device()
    # init distributed env first, since logger depends on the dist info.
    if mmcfg.launcher == "none":
        distributed = False
    else:
        distributed = True
        init_dist(mmcfg.launcher, **mmcfg.dist_params)

    test_dataloader_default_args = dict(
        samples_per_gpu=1, workers_per_gpu=2, dist=distributed, shuffle=False
    )

    # in case the test dataset is concatenated
    if isinstance(mmcfg.data.test, dict):
        mmcfg.data.test.test_mode = True
        if mmcfg.data.test_dataloader.get("samples_per_gpu", 1) > 1:
            # Replace 'ImageToTensor' to 'DefaultFormatBundle'
            mmcfg.data.test.pipeline = replace_ImageToTensor(mmcfg.data.test.pipeline)
    elif isinstance(mmcfg.data.test, list):
        for ds_cfg in mmcfg.data.test:
            ds_cfg.test_mode = True
        if cfg.data.test_dataloader.get("samples_per_gpu", 1) > 1:
            for ds_cfg in mmcfg.data.test:
                ds_cfg.pipeline = replace_ImageToTensor(ds_cfg.pipeline)
    test_loader_cfg = {
        **test_dataloader_default_args,
        **mmcfg.data.get("test_dataloader", {}),
    }

    rank, _ = get_dist_info()
    # allows not to create
    if mmcfg.tmpdir is not None and rank == 0:
        mmcv.mkdir_or_exist(osp.abspath(mmcfg.tmpdir))
        timestamp = time.strftime("%Y%m%d_%H%M%S", time.localtime())
        json_file = osp.join(mmcfg.tmpdir, f"eval_{timestamp}.json")

    # build the dataloader
    dataset = build_dataset(mmcfg.data.test)
    data_loader = build_dataloader(dataset, **test_loader_cfg)

    # build the model and load checkpoint
    mmcfg.model.train_cfg = None
    model = build_detector(mmcfg.model, test_cfg=mmcfg.get("test_cfg"))
    fp16_cfg = mmcfg.get("fp16", None)
    if fp16_cfg is not None:
        wrap_fp16_model(model)
    checkpoint = load_checkpoint(model, mmcfg.checkpoint, map_location="cpu")
    if mmcfg.fuse_conv_bn:
        model = fuse_conv_bn(model)
    # old versions did not save class info in checkpoints, this walkaround is
    # for backward compatibility
    if "CLASSES" in checkpoint.get("meta", {}):
        model.CLASSES = checkpoint["meta"]["CLASSES"]
        CLASSES = checkpoint["meta"]["CLASSES"]
    else:
        model.CLASSES = dataset.CLASSES
        CLASSES = dataset.CLASSES

    if not distributed:
        model = build_dp(model, mmcfg.device, device_ids=mmcfg.gpu_ids)
        outputs = single_gpu_test(
            model, data_loader, mmcfg.show, mmcfg.show_dir, mmcfg.show_score_thr
        )
    else:
        model = build_ddp(
            model,
            mmcfg.device,
            device_ids=[int(os.environ["LOCAL_RANK"])],
            broadcast_buffers=False,
        )
        outputs = multi_gpu_test(
            model,
            data_loader,
            mmcfg.tmpdir,
            mmcfg.gpu_collect or mmcfg.evaluation.get("gpu_collect", False),
        )

    # logger.debug(outputs)
    rank, _ = get_dist_info()
    if rank == 0:
        if mmcfg.result_out:
            logger.info(f"\nwriting results to {mmcfg.result_out}")
            det_df = []
            for i, out in enumerate(outputs):
                im_path = Path(dataset.data_infos[i]["file_name"])
                curr_dict = {"filename": im_path.stem, "ext": im_path.suffix}
                for cat, bbox in enumerate(out):
                    curr_dict.update({f"det_{CLASSES[cat]}": bbox})
                det_df.append(curr_dict)
            det_df = pd.DataFrame(det_df)
            det_df.to_pickle(mmcfg.result_out)
            # mmcv.dump(outputs, mmcfg.result_out)

            # Create gt df
            gt_df = []
            for i in range(len(dataset)):
                im_path = Path(dataset.data_infos[i]["file_name"])
                imageHeight = dataset.data_infos[i]["height"]
                imageWidth = dataset.data_infos[i]["width"]
                curr_dict = {
                    "filename": im_path.stem,
                    "imageHeight": imageHeight,
                    "imageWidth": imageWidth,
                }
                for l in CLASSES:
                    curr_dict.update({f"real_{l}": []})
                label_data = dataset.get_ann_info(i)
                # get bbox and label info
                for l, box in zip(label_data["labels"], label_data["bboxes"]):
                    curr_label = f"real_{CLASSES[l]}"
                    box = box.astype(int).tolist()
                    curr_dict[curr_label].append(box)
                gt_df.append(curr_dict)
            gt_df = pd.DataFrame(gt_df)

            # Merge df
            merged_df = pd.merge(left=det_df, right=gt_df, on="filename", how="left")

            # Customized evaluation
            score_thr = mmcfg.model.test_cfg.score_thr
            miou_df = []
            for _, row in merged_df.iterrows():
                curr_dict = {"filename": row.filename}
                sum_iou = 0
                divisor = 0
                for label in checkpoint["meta"]["CLASSES"]:
                    curr_det = row[f"det_{label}"]
                    curr_det = curr_det[np.where(curr_det[:, -1] > score_thr)]
                    iou, _ = calculate_iou(
                        curr_det, row[f"real_{label}"], row.imageHeight, row.imageWidth
                    )
                    curr_dict.update({f"{label}_mIoU": iou})
                    if iou != -1:
                        sum_iou += iou
                        divisor += 1

                if divisor == 0:
                    curr_dict.update({"total_mIoU": -1})
                else:
                    curr_dict.update({"total_mIoU": sum_iou / divisor})
                miou_df.append(curr_dict)
            miou_df = pd.DataFrame(miou_df)
            miou = np.array(miou_df.total_mIoU)[np.where(miou_df.total_mIoU != -1)]
            miou = round(np.mean(miou), 2)
            merged_df = pd.merge(merged_df, miou_df, how="left", on="filename")

            # Save raw data
            save_path = Path(mmcfg.tmpdir) / "combined_data.pkl"
            logger.info(f"Save evaluation result at {save_path}")
            merged_df.to_pickle(save_path)
            tracking.log_model(save_path)

        kwargs = {} if mmcfg.eval_options is None else mmcfg.eval_options
        if mmcfg.format_only:
            dataset.format_results(outputs, **kwargs)
        if mmcfg.eval:
            eval_kwargs = cfg.get("evaluation", {}).copy()
            # hard-code way to remove EvalHook args
            for key in [
                "interval",
                "tmpdir",
                "start",
                "gpu_collect",
                "save_best",
                "rule",
                "dynamic_intervals",
            ]:
                eval_kwargs.pop(key, None)
            eval_kwargs.update(dict(metric=mmcfg.eval, **kwargs))
            metric = dataset.evaluate(outputs, **eval_kwargs)
            print(metric)
            metric_dict = dict(config=mmcfg.dump(), metric=metric)
            if mmcfg.tmpdir is not None and rank == 0:
                mmcv.dump(metric_dict, json_file)

    logger.info(f"Model mIoU => {miou}")
    tracking.log_outputs(
        mIoU="{:0.2f}".format(miou),
    )
    logger.info("All Evaluation Done.")


def calculate_iou(det_boxes, real_boxes, img_h, img_w):
    if len(det_boxes) == 0 and len(real_boxes) == 0:
        return -1, []
    elif len(det_boxes) == 0 or len(real_boxes) == 0:
        return 0, []

    # Draw detection boxes at a blank image
    det_area = []
    det_merged = np.zeros((img_h, img_w), dtype=np.uint8)
    for det in det_boxes:
        xmin, ymin, xmax, ymax = [int(x) for x in det[:4]]
        xmin = max(xmin, 0)
        ymin = max(ymin, 0)
        xmax = min(xmax, img_w)
        ymax = min(ymax, img_h)
        det_area.append((xmax - xmin) * (ymax - ymin))
        det_merged[ymin:ymax, xmin:xmax] = 1

    # Draw ground truth boxes at a blank image
    gt_merged = np.zeros((img_h, img_w), dtype=np.uint8)
    for gt in real_boxes:
        xmin, ymin, xmax, ymax = [int(x) for x in gt[:4]]
        xmin = max(xmin, 0)
        ymin = max(ymin, 0)
        xmax = min(xmax, img_w)
        ymax = min(ymax, img_h)
        gt_merged[ymin:ymax, xmin:xmax] = 1

    # Add two masks
    all_merged = det_merged + gt_merged
    # The Area of elements bigger than 0 means union of detection and ground truth
    union = np.sum(all_merged[np.where(all_merged > 0)])
    # The Area of elements bigger than 1 means intersection between
    # detection and ground truth
    inter = np.sum(all_merged[np.where(all_merged > 1)])

    return inter / union, det_area


if __name__ == "__main__":
    run_app()
