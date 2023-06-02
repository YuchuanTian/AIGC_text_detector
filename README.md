# Multiscale Positive-Unlabeled Detection of AI-Generated Texts

*Yuchuan Tian, Hanting Chen, Xutao Wang, Zheyuan Bai, Qinghua Zhang, Ruifeng Li, Chao Xu, Yunhe Wang*

This is the code for Multiscale Positive-Unlabeled Detection of AI-Generated Texts.

ArXiv: https://arxiv.org/abs/2305.18149



## Preparation

Install requirement packages:

```shell
pip install -r requirements.txt
```

Download datasets to directory: ```./data``` 

Download nltk package punct (This step could be done by ```nltk``` api: ```nltk.download('punkt')```)

Download pretrained models (This step could be automatically done by ```transformers```)

The link for the HC3 dataset: https://github.com/Hello-SimpleAI/chatgpt-comparison-detection/blob/main/HC3/README.md


Before running, the directory should contain the following files:

```
├── data
│   ├── unfilter_full
│   │   ├── en_test.csv
│   │   └── en_train.csv
│   └── unfilter_sent
│       ├── en_test.csv
│       └── en_train.csv
├── README.md
├── corpus_cleaning_kit.py
├── dataset.py
├── multiscale_kit.py
├── option.py
├── pu_loss_mod.py
├── prior_kit.py
├── requirements.txt
├── train.py
└── utils.py
```



## Running

The script for running is ```train.py```.

#### RoBERTa on HC3

Commands for seed=0,1,2:

```shell
CUDA_VISIBLE_DEVICES=0 python train.py --batch-size 32 --max-sequence-length 512 --train-data-file unfilter_full/en_train.csv --val-data-file unfilter_full/en_test.csv --model-name roberta-base --local-data data --lamb 0.4 --prior 0.2 --pu_type dual_softmax_dyn_dtrun --len_thres 55 --aug_min_length 1 --max-epochs 1 --weight-decay 0 --mode original_single --aug_mode sentence_deletion-0.25 --clean 1 --val_file1 unfilter_sent/en_test.csv --quick_val 1 --learning-rate 5e-05 --seed 0

CUDA_VISIBLE_DEVICES=0 python train.py --batch-size 32 --max-sequence-length 512 --train-data-file unfilter_full/en_train.csv --val-data-file unfilter_full/en_test.csv --model-name roberta-base --local-data data --lamb 0.4 --prior 0.2 --pu_type dual_softmax_dyn_dtrun --len_thres 55 --aug_min_length 1 --max-epochs 1 --weight-decay 0 --mode original_single --aug_mode sentence_deletion-0.25 --clean 1 --val_file1 unfilter_sent/en_test.csv --quick_val 1 --learning-rate 5e-05 --seed 1

CUDA_VISIBLE_DEVICES=0 python train.py --batch-size 32 --max-sequence-length 512 --train-data-file unfilter_full/en_train.csv --val-data-file unfilter_full/en_test.csv --model-name roberta-base --local-data data --lamb 0.4 --prior 0.2 --pu_type dual_softmax_dyn_dtrun --len_thres 55 --aug_min_length 1 --max-epochs 1 --weight-decay 0 --mode original_single --aug_mode sentence_deletion-0.25 --clean 1 --val_file1 unfilter_sent/en_test.csv --quick_val 1 --learning-rate 5e-05 --seed 2

```

#### BERT on HC3

Commands for seed=0,1,2:

```shell

CUDA_VISIBLE_DEVICES=0 python train.py --batch-size 32 --max-sequence-length 512 --train-data-file unfilter_full/en_train.csv --val-data-file unfilter_full/en_test.csv --model-name bert-base-cased --local-data data --lamb 0.5 --prior 0.3 --pu_type dual_softmax_dyn_dtrun --len_thres 60 --aug_min_length 1 --max-epochs 1 --weight-decay 0 --mode original_single --aug_mode sentence_deletion-0.25 --clean 1 --val_file1 unfilter_sent/en_test.csv --quick_val 1 --learning-rate 5e-05 --seed 0


CUDA_VISIBLE_DEVICES=0 python train.py --batch-size 32 --max-sequence-length 512 --train-data-file unfilter_full/en_train.csv --val-data-file unfilter_full/en_test.csv --model-name bert-base-cased --local-data data --lamb 0.5 --prior 0.3 --pu_type dual_softmax_dyn_dtrun --len_thres 60 --aug_min_length 1 --max-epochs 1 --weight-decay 0 --mode original_single --aug_mode sentence_deletion-0.25 --clean 1 --val_file1 unfilter_sent/en_test.csv --quick_val 1 --learning-rate 5e-05 --seed 1


CUDA_VISIBLE_DEVICES=0 python train.py --batch-size 32 --max-sequence-length 512 --train-data-file unfilter_full/en_train.csv --val-data-file unfilter_full/en_test.csv --model-name bert-base-cased --local-data data --lamb 0.5 --prior 0.3 --pu_type dual_softmax_dyn_dtrun --len_thres 60 --aug_min_length 1 --max-epochs 1 --weight-decay 0 --mode original_single --aug_mode sentence_deletion-0.25 --clean 1 --val_file1 unfilter_sent/en_test.csv --quick_val 1 --learning-rate 5e-05 --seed 2

```

## Acknowledgement

Our code refers to the following GitHub repo:

https://github.com/openai/gpt-2-output-dataset

We sincerely thank their authors for open-sourcing.

## Future Work

A demo is to be deployed. It will be released later. Stay tuned!
## Discussion

You are welcomed to join the WeChat Discussion Group by scanning the QR code below:

<img src="imgs/QR.jpg" alt="QR" style="zoom:35%;" />