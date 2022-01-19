# CUBOX 유효성 검증 소스 코드


### 학습, 평가 수행 환경 설정
- 소스 코드 다운로드 (위치: PATH_TO_CODE)
    ```bash
    git clone https://github.com/Yuuraa/CUBOX_final --recurse-submodules
    ```
- 이미지 명: yoorachoi/cubox:beit-resnet
- 명령어
    ```bash
    nvidia-docker run -it --gpus all --ipc host \
            --mount type="bind",source=[PATH_TO_DATASET],target="/dataset" \
            --mount type="bind",source=[PATH_TO_CODE],target="/cubox"\
            yoorachoi/cubox:beit-resnet /bin/bash
    ```
</br>

### Docker container 내 소스 코드 구조
**코드 구조**
```bash
/cubox/BEiT-CUBOX
├── backbone
|   └── beit.py
├── configs
|   ├── _base_
|   |    ├── datasets
|   |    |    ├── cubox.py
|   |    |    ├── cubox_map.py
|   |    |    └── ...
|   |    ├── models
|   |    |    ├── upernet_beit_cubox.py
|   |    |    └── upernet_beit_cubox_map.py
|   |    └── schedules
|   |    |    ├── schedule_160k.py
|   |    |    └── schedule_320k.py
|   ├── beit/upernet
|   |    ├── upernet_beit_base_12_256_slide_160k_ade20k_pt2ft.py
|   |    └── upernet_beit_base_12_256_slide_160k_ade20k_pt2ft_map.py
|   └── test_configs
|        ├── test_all.py
|        ├── test_none.py
|        └── ...
├── mmcv_custom
├── tools
└── total_all2all_wo_pretrained_32 #(checkpoint dir)

/cubox/CUBOX_classification
├── classification
|   ├── finetune
|   |    ├── __init_.py
|   |    └── train.py # train
|   ├── __init__.py
|   └── eval.py # validation
├── dataset
|   ├── __init__.py
|   ├── cubox.py # CUBOX Dataset
|   ├── preprocess.py # data preprocessing
|   └── transforms.py # data transformation
├── models
|   └── __init__.py
├── utils
|   ├── __init__.py
|   ├── eval_meter.py
|   ├── logging.py
|   ├── observe.py # 평가 지표들 합산
|   ├── save.py # 학습 체크포인트 저장
|   └── util.py
├── configs.py
├── main.py # 학습, 검증
├── inference_log.py # 평가 지표 로깅
└── saved_results/finetune/all2all/basic/20220109-165155 # (checkpoint dir)
```
**Dataset Folder**
```bash
/dataset
├── images
|   ├── train
|   |   ├── none
|   |   |   ├── class1
|   |   |   |   ├── image1
|   |   |   |   └── image2
|   |   |   |   └── ...
|   |   |   └── class2
|   |   |   └── class3
|   |   |   └── ...
|   |   └── semitransparent
|   |   |   ├── class1
|   |   |   └── class2
|   |   |   └── class3
|   |   |   └── ...
|   |   └── wiredense
|   |   |   ├── class1
|   |   |   └── class2
|   |   |   └── class3
|   |   |   └── ...
|   |   └── wireloose
|   |   |   ├── class1
|   |   |   └── class2
|   |   |   └── class3
|   |   |   └── ...
|   |   └── wiremedium
|   |   |   ├── class1
|   |   |   └── class2
|   |   |   └── class3
|   |   |   └── ...
|   └── validation # same structure as train
|   └── test        # same structure as train
|
└──  seg_map          # same structure as images
```


## Classification
### Train (from scratch)
만약 resnet50 pretrained weight를 로딩하고 싶다면 not_pretrain 인자를 제거하면 됨
- 실험 스크립트
    ```bash
    CUDA_VISIBLE_DEVICES=1 python3 main.py -b 64 --method finetune --data_config all2all --not_pretrain --experiment total_wo_pretrained --data_root /dataset/images
    ```
- 의미
    ```bash
    CUDA_VISIBLE_DEVICES=1 python3 main.py -b [BATCH_SIZE] --method finetune --data_config [TRAINSET,TESTSET] --not_pretrain --experiment [EXPERIMENT_NAME] --data_root /dataset/images
    ```

### Test
- 실험 스크립트
    ```bash
    CUDA_VISIBLE_DEVICES=0 python3 inference_log.py --ckpt_dir ./saved_results/finetune/all2all/basic/20220109-165155 --data_config all2all --data_root /dataset/images > eval_result.txt
    ```
- 의미
    ```bash
    CUDA_VISIBLE_DEVICES=0 python3 inference_log.py --ckpt_dir [CHECKPOINT_PATH] --data_config all2all --data_root /dataset/images > [LOG_OUTPUT_FILE]
    ```


## Segmentation
### Train (from scratch)
- 실험 스크립트
    ```bash
    bash tools/dist_train.sh configs/beit/upernet/upernet_beit_base_12_256_slide_160k_cubox_pt2ft_map.py 4 --work-dir /cubox/BEiT-CUBOX/total_all2all_wo_pretrained_32 --seed 0 --deterministic
    ```
- 의미
    ```bash
    bash tools/dist_train.sh configs/beit/upernet/upernet_beit_base_12_256_slide_160k_cubox_pt2ft_map.py [NUM_GPUS] --work-dir [WORKDIR] --seed 0 --deterministic
    ```
    - WORKDIR: 모델의 checkpoint, 학습 config 등이 저장되는 경로

### Test
- 실험 스크립트
    ```bash
    bash tools/dist_test.sh configs/beit/upernet/upernet_beit_base_12_256_slide_160k_cubox_pt2ft_map.py total_all2all_wo_pretrained_32/iter_160000.pth 4 --eval mAP    
    ```
- 의미
    ```bash
    bash tools/dist_test.sh configs/beit/upernet/upernet_beit_base_12_256_slide_160k_cubox_pt2ft_map.py [PATH_TO_CKPT] [NUM_GPUS] --eval mAP
    ```


## 결과물
- results 폴더
### Classification
- ROC Curve per Class: roc_curves
- Accuracy, AUC, F1 Score Evaluation Result: eval_result.txt

### Segmentation
- Precision-Recall Curve per Class: precision_recall_curves
- mAP Evaluation Result: result_map.txt