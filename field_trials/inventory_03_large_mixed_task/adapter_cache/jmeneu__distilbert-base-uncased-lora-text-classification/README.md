---
license: apache-2.0
library_name: peft
widget:
- text: "It was good."
  example_title: "Positive"
- text: "Not a fan, don't recommed."
  example_title: "Negative"
- text: "Better than the first one."
  example_title: "Positive"
- text: "This is not worth watching even once. "
  example_title: "Negative"
- text: "This one is a pass."
  example_title: "Positive"
tags:
- sentiment
datasets:
- imdb
metrics:
- accuracy
base_model: distilbert-base-uncased
model-index:
- name: distilbert-base-uncased-lora-text-classification
  results: []
---

<!-- This model card has been generated automatically according to the information the Trainer had access to. You
should probably proofread and complete it, then remove this comment. -->

# distilbert-base-uncased-lora-text-classification

This model is a fine-tuned version of [distilbert-base-uncased](https://huggingface.co/distilbert-base-uncased) on the imdb dataset.
It achieves the following results on the evaluation set:
- Loss: 0.4297
- Accuracy: {'accuracy': 0.86}

## Model description

More information needed

## Intended uses & limitations

More information needed

## Training and evaluation data

More information needed

## Training procedure

### Training hyperparameters

The following hyperparameters were used during training:
- learning_rate: 0.01
- train_batch_size: 4
- eval_batch_size: 4
- seed: 42
- optimizer: Adam with betas=(0.9,0.999) and epsilon=1e-08
- lr_scheduler_type: constant
- lr_scheduler_warmup_ratio: 0.1
- num_epochs: 3

### Training results

| Training Loss | Epoch | Step  | Validation Loss | Accuracy             |
|:-------------:|:-----:|:-----:|:---------------:|:--------------------:|
| 0.7852        | 1.0   | 5000  | 0.4518          | {'accuracy': 0.8486} |
| 0.6289        | 2.0   | 10000 | 0.4492          | {'accuracy': 0.8528} |
| 0.0503        | 3.0   | 15000 | 0.4297          | {'accuracy': 0.86}   |


### Framework versions

- Transformers 4.35.2
- Pytorch 2.1.0+cu118
- Datasets 2.15.0
- Tokenizers 0.15.0
## Training procedure


### Framework versions


- PEFT 0.6.2
