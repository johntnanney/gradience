---
library_name: peft
license: apache-2.0
base_model: distilbert-base-uncased
tags:
- base_model:adapter:distilbert-base-uncased
- lora
- transformers
metrics:
- accuracy
model-index:
- name: distilbert-base-uncased-lora-text-classification
  results: []
---

<!-- This model card has been generated automatically according to the information the Trainer had access to. You
should probably proofread and complete it, then remove this comment. -->

# distilbert-base-uncased-lora-text-classification

This model is a fine-tuned version of [distilbert-base-uncased](https://huggingface.co/distilbert-base-uncased) on an unknown dataset.
It achieves the following results on the evaluation set:
- Loss: 0.9216
- Accuracy: {'accuracy': 0.887}

## Model description

More information needed

## Intended uses & limitations

More information needed

## Training and evaluation data

More information needed

## Training procedure

### Training hyperparameters

The following hyperparameters were used during training:
- learning_rate: 0.001
- train_batch_size: 4
- eval_batch_size: 4
- seed: 42
- optimizer: Use OptimizerNames.ADAMW_TORCH_FUSED with betas=(0.9,0.999) and epsilon=1e-08 and optimizer_args=No additional optimizer arguments
- lr_scheduler_type: linear
- num_epochs: 10

### Training results

| Training Loss | Epoch | Step | Validation Loss | Accuracy            |
|:-------------:|:-----:|:----:|:---------------:|:-------------------:|
| No log        | 1.0   | 250  | 0.4480          | {'accuracy': 0.863} |
| 0.4694        | 2.0   | 500  | 0.3601          | {'accuracy': 0.896} |
| 0.4694        | 3.0   | 750  | 0.5534          | {'accuracy': 0.884} |
| 0.2308        | 4.0   | 1000 | 0.5800          | {'accuracy': 0.891} |
| 0.2308        | 5.0   | 1250 | 0.7042          | {'accuracy': 0.873} |
| 0.0767        | 6.0   | 1500 | 0.7120          | {'accuracy': 0.897} |
| 0.0767        | 7.0   | 1750 | 0.8353          | {'accuracy': 0.889} |
| 0.0124        | 8.0   | 2000 | 0.8839          | {'accuracy': 0.884} |
| 0.0124        | 9.0   | 2250 | 0.9220          | {'accuracy': 0.888} |
| 0.0059        | 10.0  | 2500 | 0.9216          | {'accuracy': 0.887} |


### Framework versions

- PEFT 0.18.1
- Transformers 5.0.0
- Pytorch 2.10.0+cu128
- Datasets 4.0.0
- Tokenizers 0.22.2