---
license: apache-2.0
tags:
- generated_from_trainer
datasets:
- glue
metrics:
- accuracy
model-index:
- name: distilbert-base-uncased-finetuned-LoRA-SST2
  results:
  - task:
      name: Text Classification
      type: text-classification
    dataset:
      name: glue
      type: glue
      args: sst2
    metrics:
    - name: Accuracy
      type: accuracy
      value: 0.8979357798165137
---

<!-- This model card has been generated automatically according to the information the Trainer had access to. You
should probably proofread and complete it, then remove this comment. -->

# distilbert-base-uncased-finetuned-lora-sst2

This model is a fine-tuned version of [distilbert-base-uncased](https://huggingface.co/distilbert/distilbert-base-uncased) on the glue dataset.
It achieves the following results on the evaluation set:
- Accuracy: 0.8979
- trainable model parameters: 887042
- all model parameters: 67842052
- percentage of trainable model parameters: 1.31%

## Training procedure

### Training hyperparameters

The following hyperparameters were used during training:
- learning_rate: 5e-04
- train_batch_size: 32
- eval_batch_size: 32
- seed: 42
- weight_decay: 0.01
- rank: 16
- lora_alpha: 32
- lora_dropout: 0.05
- num_epochs: 4