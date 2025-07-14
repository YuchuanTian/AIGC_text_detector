# Multiscale Positive-Unlabeled Detection of AI-Generated Texts

<p align="left">
<a href="https://arxiv.org/abs/2305.18149" alt="arXiv">
    <img src="https://img.shields.io/badge/arXiv-2305.18149-b31b1b.svg?style=flat" /></a>
<a href="https://arxiv.org/pdf/2305.18149" alt="arXiv">
    <img src="https://img.shields.io/badge/Paper-PDF-b31b1b.svg?style=flat" /></a>
<a href="https://huggingface.co/spaces/yuchuantian/AIGC_text_detector" alt="Hugging Face Models">
    <img src="https://img.shields.io/badge/%F0%9F%A4%97%20Hugging%20Face-Space-blue" /></a>
<a href="https://github.com/YuchuanTian/AIGC_text_detector/blob/main/imgs/QR.jpg"><img src="https://img.shields.io/badge/微信-二维码加群-green?logo=wechat&amp"></a>
</p>  

This is the official repo of our **ICLR'24 Spotlight** paper "Multiscale Positive-Unlabeled Detection of AI-Generated Texts".

## News

**6/25/2025 🔥🔥 [MAJOR UPDATE](#update-anchor)**: We release advanced (v3) detectors targeting latest LLMs, including DeepSeek-V3, GPT-4, and reasoning models. [En-v3](https://huggingface.co/yuchuantian/AIGC_detector_env3) / [En-v3-short](https://huggingface.co/yuchuantian/AIGC_detector_env3short) /  [Zh-v3](https://huggingface.co/yuchuantian/AIGC_detector_zhv3) /  [Zh-v3-short](https://huggingface.co/yuchuantian/AIGC_detector_zhv3short) / [DEMO](https://huggingface.co/spaces/yuchuantian/AIGC_text_detector)

**6/1/2025**: We release a beta version targeting latest LLMs, including DeepSeek-V3 and GPT-4. [En-beta](https://huggingface.co/yuchuantian/AIGC_detector_enbeta)  [Zh-beta](https://huggingface.co/yuchuantian/AIGC_detector_zhbeta)

**3/25/2025**: We release a demo (with both English and Chinese) on HuggingFace.

**3/6/2025**: We will update a brand-new detector version to align with the latest LLMs. Please keep tuned!


*BibTex* formatted citation:

```
@misc{tian2023multiscale,
      title={Multiscale Positive-Unlabeled Detection of AI-Generated Texts}, 
      author={Yuchuan Tian and Hanting Chen and Xutao Wang and Zheyuan Bai and Qinghua Zhang and Ruifeng Li and Chao Xu and Yunhe Wang},
      year={2023},
      eprint={2305.18149},
      archivePrefix={arXiv},
      primaryClass={cs.CL}
}
```

## Detector Models

We have open-sourced detector models in the paper as follows.

Links for Detectors: [Google Drive](https://drive.google.com/drive/folders/1Q_78qoaAuO8HOtt-SawimiXhli6y0Uii?usp=drive_link)  [Baidu Disk (PIN:1234)](https://pan.baidu.com/s/11hOpOxImAh1ZfDy9F5jC1Q)

We have also uploaded detector models to HuggingFace, where easy-to-use **DEMOs** and online **APIs** are provided.

| Variants                                                     | HC3-Full-En        | HC3-Sent-En        |
| ------------------------------------------------------------ | ------------------ | ------------------ |
| seed0                                                        | 98.68              | 82.84              |
| seed1 [HuggingFace: en v1](https://huggingface.co/yuchuantian/AIGC_detector_env1) | 98.56              | 87.06              |
| seed2                                                        | 97.97              | 86.02              |
| **Avg.**                                                     | **98.40$\pm$0.31** | **85.31$\pm$1.80** |

## Stronger Detectors

We have also open-sourced detector models with strengthened training strategies. Specifically, we develop a strong Chinese detector ```AIGC_detector_zhv2```, which demonstrates similar performance to SOTA closed-source Chinese detectors on various texts, including news articles, poetry, essays, etc. The **DEMOs** and **APIs** are available on HuggingFace.

| Detector                  | Google Drive                                                 | Baidu Disk                                                   | HuggingFace Link                                             |
| ------------------------- | ------------------------------------------------------------ | ------------------------------------------------------------ | ------------------------------------------------------------ |
| English, version 2 (env2) | [Google Drive](https://drive.google.com/drive/folders/11ROLjxopgx44YT9RS8RmchdxR6Yi-CJk?usp=drive_link) | [Baidu Disk (PIN:1234)](https://pan.baidu.com/s/11CQaWzsT7a-IgceOBpmb7g) | [en v2](https://huggingface.co/yuchuantian/AIGC_detector_env2) |
| Chinese, version 2 (zhv2) | [Google Drive](https://drive.google.com/drive/folders/1-a7n-T9Z1_EIWbvip2eC0ssx5rih8pQI?usp=drive_link) | [Baidu Disk (PIN:1234)](https://pan.baidu.com/s/1VPGYtswC1GJXESWzne4RPA) | [zh v2](https://huggingface.co/yuchuantian/AIGC_detector_zhv2) |

## About the Dataset

Here we provide the official link for the HC3 dataset: [Dataset Link](https://github.com/Hello-SimpleAI/chatgpt-comparison-detection/blob/main/HC3/README.md). We also provide identical dataset copies on [Google Drive](https://drive.google.com/drive/folders/10GsKfCWe_BHLdGnfhSV-_k0-PH31_oWn?usp=drive_link) and [Baidu Disk (PIN:1234)](https://pan.baidu.com/s/1OUJbPYbC2ZUAt50MFDdHrQ) for your ease of use. We acknowledge the marvelous work by HC3 authors.

#### Data Preprocessing

In Appendix B of [our paper](https://arxiv.org/pdf/2305.18149.pdf), we proposed the removal of redundant spaces in human texts of the HC3-English dataset. We have provided a helper function ```en_cleaning``` in ```corpus_cleaning_kit.py``` that takes a sentence string as input and returns a preprocessed sentence without redundant spaces.

Here we provide a cleaned version of HC3-English. In this version, all answers are cleaned (*i. e.* redundant spaces are removed). However, please use the original version of HC3 for all experiments in our paper, as we have embedded the cleaning procedures in the training & validation scripts.

**CLEANED** HC3-English Link:     [Google Drive](https://drive.google.com/drive/folders/11m9w7blNjUR2VE5N5AU7aOmj9YZhOyLy?usp=drive_link)    [Baidu Disk (PIN:1234)](https://pan.baidu.com/s/1kKSiyj1Nv2me6mODZd0Y4A)

##  Preparation

- Install requirement packages:

```shell
pip install -r requirements.txt
```

- Download datasets to directory: ```./data``` 

- Download nltk package punct (This step could be done by ```nltk``` api: ```nltk.download('punkt')```)

- Download pretrained models (This step could be automatically done by ```transformers```)


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

## Training

The script for training is ```train.py```.

#### RoBERTa on HC3-English

Commands for seed=0,1,2:

```shell
CUDA_VISIBLE_DEVICES=0 python train.py --batch-size 32 --max-sequence-length 512 --train-data-file unfilter_full/en_train.csv --val-data-file unfilter_full/en_test.csv --model-name roberta-base --local-data data --lamb 0.4 --prior 0.2 --pu_type dual_softmax_dyn_dtrun --len_thres 55 --aug_min_length 1 --max-epochs 1 --weight-decay 0 --mode original_single --aug_mode sentence_deletion-0.25 --clean 1 --val_file1 unfilter_sent/en_test.csv --quick_val 1 --learning-rate 5e-05 --seed 0

CUDA_VISIBLE_DEVICES=0 python train.py --batch-size 32 --max-sequence-length 512 --train-data-file unfilter_full/en_train.csv --val-data-file unfilter_full/en_test.csv --model-name roberta-base --local-data data --lamb 0.4 --prior 0.2 --pu_type dual_softmax_dyn_dtrun --len_thres 55 --aug_min_length 1 --max-epochs 1 --weight-decay 0 --mode original_single --aug_mode sentence_deletion-0.25 --clean 1 --val_file1 unfilter_sent/en_test.csv --quick_val 1 --learning-rate 5e-05 --seed 1

CUDA_VISIBLE_DEVICES=0 python train.py --batch-size 32 --max-sequence-length 512 --train-data-file unfilter_full/en_train.csv --val-data-file unfilter_full/en_test.csv --model-name roberta-base --local-data data --lamb 0.4 --prior 0.2 --pu_type dual_softmax_dyn_dtrun --len_thres 55 --aug_min_length 1 --max-epochs 1 --weight-decay 0 --mode original_single --aug_mode sentence_deletion-0.25 --clean 1 --val_file1 unfilter_sent/en_test.csv --quick_val 1 --learning-rate 5e-05 --seed 2

```

#### BERT on HC3-English

Commands for seed=0,1,2:

```shell
CUDA_VISIBLE_DEVICES=0 python train.py --batch-size 32 --max-sequence-length 512 --train-data-file unfilter_full/en_train.csv --val-data-file unfilter_full/en_test.csv --model-name bert-base-cased --local-data data --lamb 0.5 --prior 0.3 --pu_type dual_softmax_dyn_dtrun --len_thres 60 --aug_min_length 1 --max-epochs 1 --weight-decay 0 --mode original_single --aug_mode sentence_deletion-0.25 --clean 1 --val_file1 unfilter_sent/en_test.csv --quick_val 1 --learning-rate 5e-05 --seed 0


CUDA_VISIBLE_DEVICES=0 python train.py --batch-size 32 --max-sequence-length 512 --train-data-file unfilter_full/en_train.csv --val-data-file unfilter_full/en_test.csv --model-name bert-base-cased --local-data data --lamb 0.5 --prior 0.3 --pu_type dual_softmax_dyn_dtrun --len_thres 60 --aug_min_length 1 --max-epochs 1 --weight-decay 0 --mode original_single --aug_mode sentence_deletion-0.25 --clean 1 --val_file1 unfilter_sent/en_test.csv --quick_val 1 --learning-rate 5e-05 --seed 1


CUDA_VISIBLE_DEVICES=0 python train.py --batch-size 32 --max-sequence-length 512 --train-data-file unfilter_full/en_train.csv --val-data-file unfilter_full/en_test.csv --model-name bert-base-cased --local-data data --lamb 0.5 --prior 0.3 --pu_type dual_softmax_dyn_dtrun --len_thres 60 --aug_min_length 1 --max-epochs 1 --weight-decay 0 --mode original_single --aug_mode sentence_deletion-0.25 --clean 1 --val_file1 unfilter_sent/en_test.csv --quick_val 1 --learning-rate 5e-05 --seed 2

```



## Updates

<a name="update-anchor"></a>

A major update (v3) is ongoing, with updated English and Chinese detectors, datasets, and codes. This time, our detector targets at corpora from MiniMax-Text-01, Llama-3.3-70B-Instruct, Qwen2.5-72B-Instruct, DeepSeek-V3, and GPT-4-Turbo. Besides, Chain-of-Thought (CoT) models are covered, including DeepSeek-R1, GPT-o3, and QwQ-32B. Please keep tuned for future updates!


| Version                          | English                                                      | Chinese                                                      | Comments                                                     |
| -------------------------------- | ------------------------------------------------------------ | ------------------------------------------------------------ | ------------------------------------------------------------ |
| 6/25/2025 AIGC-Detector-V3       | [En_v3](https://huggingface.co/yuchuantian/AIGC_detector_env3) | [Zh_v3](https://huggingface.co/yuchuantian/AIGC_detector_zhv3) | Version 3. Strongest performance on normal AI texts. CoT LLMs are covered. |
| 6/25/2025 AIGC-Detector-V3-Short | [En_v3_short](https://huggingface.co/yuchuantian/AIGC_detector_env3short) | [Zh_v3_short](https://huggingface.co/yuchuantian/AIGC_detector_zhv3short) | Version 3 targeting shorter texts without sacrificing longer ones. CoT LLMs are covered. |
| 5/31/2025 AIGC-Detector-Beta     | [En_beta](https://huggingface.co/yuchuantian/AIGC_detector_enbeta) | [Zh_beta](https://huggingface.co/yuchuantian/AIGC_detector_zhbeta) | A beta version. Stronger versions will be released in the next update. |



## Acknowledgement

Our code refers to the following GitHub repo:

https://github.com/openai/gpt-2-output-dataset

We sincerely thank their authors for open-sourcing.
