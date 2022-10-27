# Kaflix Damage Model

## 0. 필요한 패키지 설치

실행에 필요한 패키지들은 `requirements.txt`에 작성되어 있습니다.
아래 명령을 실행하여 패키지들을 설치할 수 있습니다.

```
pip install -r requirements.txt
```

## 1. 데이터셋 및 학습 모델 정보

### 1.1. 데이터셋의 구조

데이터셋은 `dataset-root`아래 이미지가 포함된 `train`, `valid` 폴더, coco-style의 JSON파일이 있는 `coco_annotations` 폴더, 라벨링 프로그램에서 사용하는 포멧의 JSON파일이 있는 `labelme_annotations`폴더로 구성되어 있습니다.

```
dataset-root
  | - train
        | - image1.jpg
          - image2.jpg
          ...
    - valid
        | - image1-1.jpg
          - image2-1.jpg
          ...
    - coco_annotations
        | - train.json
          - valid.json
    - labelme_annotations
        | - train
              | - image1.json
                - image2.json
                ...
          - valid
              | - image1-1.json
                - image2-1.json
                ...
```

### 1.2. 학습된 모델의 위치

학습이 완료된 모델은 `./trained_model/best_ckpt.pth`에 저장되어 있습니다.
학습시 사용했던 파라미터는 `./trained_model/train_config.py`에서 확인할 수 있습니다.

## 2. 학습 및 평가

### 2.1. config 파일 구조

```yaml
sodaflow:
    dataset_path:
        - "/datasets/sample-data/"
    start_ckpt_path: "./trained_model"
    test_model_path: "./trained_model"

train_inputs:
  dataset_type: CocoDataset
  classes:
    - scratch
    - dent
    - complex_damage
    - broken
  num_classes: 0
  train_ann: "coco_annotations/train.json"
  train_imgs: train
  valid_ann: "coco_annotations/valid.json"
  valid_imgs: valid
  test_ann: "coco_annotations/valid.json"
  test_imgs: valid
  test_score_thr: 0.01
  max_epoch: 5
  lr: 0.001
  warmup_ratio: 0.1
  warmup_iters: 5000
  lr_periods: [3, 2]
  lr_restart_weights: [1.0, 0.5]
  min_lr_ratio: 0.01
  log_interval: 100
  eval_interval: 1
  ckpt_interval: 1
  max_keep_ckpts: 1
  save_best_metric: auto
  SyncNormHook_interval: 10
  FT: true
  batch_size: 1
  workers: 4
  lanucher: none
eval_inputs:
  dataset_type: CocoDataset
  test_ann: "coco_annotations/valid.json"
  test_imgs: valid
  eval: bbox # "bbox", "segm", "proposal" for COCO, and "mAP", "recall" for PASCAL VOC
  batch_size: 4
  workers: 4
  lanucher: none
  fuse_conv_bn: true
  show: false
  show_dir: "./tmp"
  show_score_thr: 0.3
  gpu_collect: false
  tmpdir: "./result_tmp"
  result_out: "./result_tmp/eval_result.pkl"
  format_only: true
  eval_options:
  test_score_thr: 0.2
  model_config: "./configs/dyhead_config.py"

log_level: INFO
experiment_name: test
work_dir: "./model_logs"
model_config: "./configs/dyhead_config.py"
```

**주요 설정값**
* sodaflow:
  * dataset_path: 학습 및 검증데이터셋의 경로
  * start_ckpt_path: 미리 학습된 모델이 있다면 그 모델이 있는 폴더의 경로를 지정
* train_inputs
  * train_ann: dataset 폴더 기준으로 coco-style 라벨링 파일의 경로
  * train_imgs: dataset 폴더 기준으로 학습 이미지가 있는 경로
  * max_epoch: 총 학습 수행 횟수
  * lr: 학습시 적용할 learning rate
  * warmup_ratio: warm up 시 적용할 learning rate 비율
  * warmup_iters: warm up을 몇번 수행할지
  * lr_periods: lr을 변경할 구간, 리스트로 작성하며 리스트 안의 숫자합은 max_epoch와 같아야함.
  * lr_restart_weights: lr을 변경시 적용할 ratio, 리스트의 길이는 lr_periods에서 지정한 리스트의 길이와 같아야함.
  * log_interval: log metric을 남길 주기
  * batch_size: 한번에 몇개의 데이터 샘플을 학습할지 지정
* work_dir: 학습된 모델이 저장될 경로

### 2.2. 학습 실행

`CUDA_VISIBLE_DEVICES`에 학습에 활용할 GPU 번호를 지정하여 실행합니다.

```
CUDA_VISIBLE_DEVICES=0 python train_main.py
```

학습이 진행되면 config 파일에서 `work_dir`로 지정한 경로에 `best_ckpt.pth` 모델 파일이 생성되며 학습이 끝나면 성능이 가장 좋은 모델이 `best_ckpt.pth`에 저장됩니다.

### 2.3. 평가 실행

#### config 파일에서 체크해야할 사항
* sodaflow:
  * dataset_path: 데이터셋 경로 확인
  * test_model_path: 학습이 완료된 모델이 있는 폴더의 경로 확인
* eval_inputs:
  * show: `true`일 경우 `show_dir`에 지정한 경로에 시각화한 결과가 저장
  * test_score_thr: float, 지정한 confidence score보다 낮은 객체는 결과에서 삭제

config파일에 대한 수정이 끝나면 `CUDA_VISIBLE_DEVICES`에 평가에 활용할 GPU 번호를 지정하여 실행합니다.

```
CUDA_VISIBLE_DEVICES=0 python evaluate.py
```

실행이 완료되면 `result_out`에 지정한 경로에 인식 결과에 대한 로그가 저장됩니다.
