---
license: apache-2.0
library_name: peft
tags:
- generated_from_trainer
metrics:
- accuracy
base_model: distilbert-base-uncased
model-index:
- name: distilbert-base-uncased-lora-imdb-sentiment
  results: []
---

<!-- This model card has been generated automatically according to the information the Trainer had access to. You
should probably proofread and complete it, then remove this comment. -->

# distilbert-base-uncased-lora-imdb-sentiment

This model is a fine-tuned version of [distilbert-base-uncased](https://huggingface.co/distilbert-base-uncased) on an unknown dataset.
It achieves the following results on the evaluation set:
- Loss: 0.6932
- Accuracy: 0.4968

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
- optimizer: Adam with betas=(0.9,0.999) and epsilon=1e-08
- lr_scheduler_type: linear
- num_epochs: 10

### Training results

| Training Loss | Epoch | Step  | Validation Loss | Accuracy |
|:-------------:|:-----:|:-----:|:---------------:|:--------:|
| 0.6937        | 1.0   | 7500  | 0.6977          | 0.5032   |
| 0.6952        | 2.0   | 15000 | 0.6931          | 0.5032   |
| 0.6933        | 3.0   | 22500 | 0.6933          | 0.4968   |
| 0.6933        | 4.0   | 30000 | 0.6931          | 0.5032   |
| 0.6932        | 5.0   | 37500 | 0.6931          | 0.5032   |
| 0.6933        | 6.0   | 45000 | 0.6932          | 0.4968   |
| 0.6931        | 7.0   | 52500 | 0.6932          | 0.4968   |
| 0.6933        | 8.0   | 60000 | 0.6931          | 0.5032   |
| 0.6931        | 9.0   | 67500 | 0.6932          | 0.4968   |
| 0.6932        | 10.0  | 75000 | 0.6932          | 0.4968   |


### Framework versions

- PEFT 0.8.2
- Transformers 4.37.2
- Pytorch 2.1.0+cu121
- Datasets 2.17.1
- Tokenizers 0.15.1