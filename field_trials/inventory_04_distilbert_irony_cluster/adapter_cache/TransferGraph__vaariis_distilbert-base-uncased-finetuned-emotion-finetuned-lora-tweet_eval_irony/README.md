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
base_model: vaariis/distilbert-base-uncased-finetuned-emotion
model-index:
- name: vaariis_distilbert-base-uncased-finetuned-emotion-finetuned-lora-tweet_eval_irony
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
      value: 0.6554973821989529
      name: accuracy
---

<!-- This model card has been generated automatically according to the information the Trainer had access to. You
should probably proofread and complete it, then remove this comment. -->

# vaariis_distilbert-base-uncased-finetuned-emotion-finetuned-lora-tweet_eval_irony

This model is a fine-tuned version of [vaariis/distilbert-base-uncased-finetuned-emotion](https://huggingface.co/vaariis/distilbert-base-uncased-finetuned-emotion) on the tweet_eval dataset.
It achieves the following results on the evaluation set:
- accuracy: 0.6555

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
| 0.5089   | None       | 0     |
| 0.5969   | 0.6972     | 0     |
| 0.6073   | 0.6437     | 1     |
| 0.6304   | 0.6129     | 2     |
| 0.6482   | 0.5793     | 3     |
| 0.6450   | 0.5575     | 4     |
| 0.6419   | 0.5356     | 5     |
| 0.6565   | 0.5183     | 6     |
| 0.6555   | 0.5068     | 7     |


### Framework versions

- PEFT 0.8.2
- Transformers 4.37.2
- Pytorch 2.2.0
- Datasets 2.16.1
- Tokenizers 0.15.2