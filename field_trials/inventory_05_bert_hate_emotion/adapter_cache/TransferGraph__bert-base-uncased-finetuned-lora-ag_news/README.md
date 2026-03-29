---
license: apache-2.0
library_name: peft
tags:
- parquet
- text-classification
datasets:
- ag_news
metrics:
- accuracy
base_model: bert-base-uncased
model-index:
- name: bert-base-uncased-finetuned-lora-ag_news
  results:
  - task:
      type: text-classification
      name: Text Classification
    dataset:
      name: ag_news
      type: ag_news
      config: default
      split: test
      args: default
    metrics:
    - type: accuracy
      value: 0.9327631578947368
      name: accuracy
---

<!-- This model card has been generated automatically according to the information the Trainer had access to. You
should probably proofread and complete it, then remove this comment. -->

# bert-base-uncased-finetuned-lora-ag_news

This model is a fine-tuned version of [bert-base-uncased](https://huggingface.co/bert-base-uncased) on the ag_news dataset.
It achieves the following results on the evaluation set:
- accuracy: 0.9328

## Model description

More information needed

## Intended uses & limitations

More information needed

## Training and evaluation data

More information needed

## Training procedure

### Training hyperparameters

The following hyperparameters were used during training:
- learning_rate: 0.0004
- train_batch_size: 24
- eval_batch_size: 24
- seed: 42
- optimizer: Adam with betas=(0.9,0.999) and epsilon=1e-08
- lr_scheduler_type: linear
- num_epochs: 4

### Training results

| accuracy | train_loss | epoch |
|:--------:|:----------:|:-----:|
| 0.2263   | None       | 0     |
| 0.9232   | 0.2921     | 0     |
| 0.9263   | 0.2171     | 1     |
| 0.9325   | 0.1989     | 2     |
| 0.9328   | 0.1889     | 3     |


### Framework versions

- PEFT 0.8.2
- Transformers 4.37.2
- Pytorch 2.2.0
- Datasets 2.16.1
- Tokenizers 0.15.2