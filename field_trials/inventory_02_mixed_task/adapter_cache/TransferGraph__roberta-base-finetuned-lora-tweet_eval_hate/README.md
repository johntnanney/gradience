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
base_model: roberta-base
model-index:
- name: roberta-base-finetuned-lora-tweet_eval_hate
  results:
  - task:
      type: text-classification
      name: Text Classification
    dataset:
      name: tweet_eval
      type: tweet_eval
      config: hate
      split: validation
      args: hate
    metrics:
    - type: accuracy
      value: 0.753
      name: accuracy
---

<!-- This model card has been generated automatically according to the information the Trainer had access to. You
should probably proofread and complete it, then remove this comment. -->

# roberta-base-finetuned-lora-tweet_eval_hate

This model is a fine-tuned version of [roberta-base](https://huggingface.co/roberta-base) on the tweet_eval dataset.
It achieves the following results on the evaluation set:
- accuracy: 0.753

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
- train_batch_size: 32
- eval_batch_size: 32
- seed: 42
- optimizer: Adam with betas=(0.9,0.999) and epsilon=1e-08
- lr_scheduler_type: linear
- num_epochs: 4

### Training results

| accuracy | train_loss | epoch |
|:--------:|:----------:|:-----:|
| 0.573    | None       | 0     |
| 0.739    | 0.5596     | 0     |
| 0.75     | 0.4559     | 1     |
| 0.744    | 0.4256     | 2     |
| 0.753    | 0.4143     | 3     |


### Framework versions

- PEFT 0.8.2
- Transformers 4.37.2
- Pytorch 2.2.0
- Datasets 2.16.1
- Tokenizers 0.15.2