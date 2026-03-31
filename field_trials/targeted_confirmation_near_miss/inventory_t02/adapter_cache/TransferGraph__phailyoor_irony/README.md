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
base_model: phailyoor/distilbert-base-uncased-finetuned-yahd
model-index:
- name: phailyoor_distilbert-base-uncased-finetuned-yahd-finetuned-lora-tweet_eval_irony
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
      value: 0.6450261780104712
      name: accuracy
---

<!-- This model card has been generated automatically according to the information the Trainer had access to. You
should probably proofread and complete it, then remove this comment. -->

# phailyoor_distilbert-base-uncased-finetuned-yahd-finetuned-lora-tweet_eval_irony

This model is a fine-tuned version of [phailyoor/distilbert-base-uncased-finetuned-yahd](https://huggingface.co/phailyoor/distilbert-base-uncased-finetuned-yahd) on the tweet_eval dataset.
It achieves the following results on the evaluation set:
- accuracy: 0.6450

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
| 0.5047   | None       | 0     |
| 0.6084   | 0.7464     | 0     |
| 0.6      | 0.6660     | 1     |
| 0.6105   | 0.6191     | 2     |
| 0.6461   | 0.5874     | 3     |
| 0.6440   | 0.5613     | 4     |
| 0.6377   | 0.5366     | 5     |
| 0.6429   | 0.5159     | 6     |
| 0.6450   | 0.4988     | 7     |


### Framework versions

- PEFT 0.8.2
- Transformers 4.37.2
- Pytorch 2.2.0
- Datasets 2.16.1
- Tokenizers 0.15.2