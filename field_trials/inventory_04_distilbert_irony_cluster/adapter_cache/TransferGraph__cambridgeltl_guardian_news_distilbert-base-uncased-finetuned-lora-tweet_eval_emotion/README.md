---
library_name: peft
tags:
- parquet
- text-classification
datasets:
- tweet_eval
metrics:
- accuracy
base_model: cambridgeltl/guardian_news_distilbert-base-uncased
model-index:
- name: cambridgeltl_guardian_news_distilbert-base-uncased-finetuned-lora-tweet_eval_emotion
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
      value: 0.7005347593582888
      name: accuracy
---

<!-- This model card has been generated automatically according to the information the Trainer had access to. You
should probably proofread and complete it, then remove this comment. -->

# cambridgeltl_guardian_news_distilbert-base-uncased-finetuned-lora-tweet_eval_emotion

This model is a fine-tuned version of [cambridgeltl/guardian_news_distilbert-base-uncased](https://huggingface.co/cambridgeltl/guardian_news_distilbert-base-uncased) on the tweet_eval dataset.
It achieves the following results on the evaluation set:
- accuracy: 0.7005

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
| 0.1551   | None       | 0     |
| 0.4439   | 1.2631     | 0     |
| 0.6150   | 1.1153     | 1     |
| 0.6898   | 0.8831     | 2     |
| 0.7005   | 0.7802     | 3     |


### Framework versions

- PEFT 0.8.2
- Transformers 4.37.2
- Pytorch 2.2.0
- Datasets 2.16.1
- Tokenizers 0.15.2