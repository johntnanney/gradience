---
license: mit
library_name: peft
tags:
- parquet
- text-classification
datasets:
- tweet_eval
metrics:
- accuracy
base_model: rmihaylov/roberta-base-sentiment-bg
model-index:
- name: rmihaylov_roberta-base-sentiment-bg-finetuned-lora-tweet_eval_irony
  results:
  - task:
      type: text-classification
      name: Text Classification
    dataset:
      name: tweet_eval
      type: tweet_eval
      config: irony
      split: validation
      args: irony
    metrics:
    - type: accuracy
      value: 0.5947643979057592
      name: accuracy
---

<!-- This model card has been generated automatically according to the information the Trainer had access to. You
should probably proofread and complete it, then remove this comment. -->

# rmihaylov_roberta-base-sentiment-bg-finetuned-lora-tweet_eval_irony

This model is a fine-tuned version of [rmihaylov/roberta-base-sentiment-bg](https://huggingface.co/rmihaylov/roberta-base-sentiment-bg) on the tweet_eval dataset.
It achieves the following results on the evaluation set:
- accuracy: 0.5948

## Model description

More information needed

## Intended uses & limitations

More information needed

## Training and evaluation data

More information needed

## Training procedure

### Training hyperparameters

The following hyperparameters were used during training:
- learning_rate: 0.0005
- train_batch_size: 32
- eval_batch_size: 32
- seed: 42
- optimizer: Adam with betas=(0.9,0.999) and epsilon=1e-08
- lr_scheduler_type: linear
- num_epochs: 8

### Training results

| accuracy | train_loss | epoch |
|:--------:|:----------:|:-----:|
| 0.5613   | None       | 0     |
| 0.5927   | 0.7209     | 0     |
| 0.5696   | 0.6881     | 1     |
| 0.5717   | 0.6710     | 2     |
| 0.6052   | 0.6587     | 3     |
| 0.5560   | 0.6512     | 4     |
| 0.5696   | 0.6427     | 5     |
| 0.5623   | 0.6344     | 6     |
| 0.5948   | 0.6325     | 7     |


### Framework versions

- PEFT 0.8.2
- Transformers 4.37.2
- Pytorch 2.2.0
- Datasets 2.16.1
- Tokenizers 0.15.2