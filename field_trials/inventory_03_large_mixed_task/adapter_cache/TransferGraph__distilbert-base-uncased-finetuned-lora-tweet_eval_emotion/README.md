---
license: apache-2.0
library_name: peft
tags:
- parquet
- text-classification
datasets:
- tweet_eval
metrics:
- accuracy
base_model: distilbert-base-uncased
model-index:
- name: distilbert-base-uncased-finetuned-lora-tweet_eval_emotion
  results:
  - task:
      type: text-classification
      name: Text Classification
    dataset:
      name: tweet_eval
      type: tweet_eval
      config: emotion
      split: validation
      args: emotion
    metrics:
    - type: accuracy
      value: 0.732620320855615
      name: accuracy
---

<!-- This model card has been generated automatically according to the information the Trainer had access to. You
should probably proofread and complete it, then remove this comment. -->

# distilbert-base-uncased-finetuned-lora-tweet_eval_emotion

This model is a fine-tuned version of [distilbert-base-uncased](https://huggingface.co/distilbert-base-uncased) on the tweet_eval dataset.
It achieves the following results on the evaluation set:
- accuracy: 0.7326

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
| 0.2353   | None       | 0     |
| 0.6631   | 1.0175     | 0     |
| 0.7139   | 0.6889     | 1     |
| 0.7246   | 0.6209     | 2     |
| 0.7326   | 0.5840     | 3     |


### Framework versions

- PEFT 0.8.2
- Transformers 4.37.2
- Pytorch 2.2.0
- Datasets 2.16.1
- Tokenizers 0.15.2