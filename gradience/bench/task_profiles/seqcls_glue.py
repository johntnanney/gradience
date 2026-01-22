"""
GLUE sequence classification task profile.
"""

from typing import Dict, Any, Tuple
from pathlib import Path

from datasets import Dataset, load_dataset
from transformers import (
    PreTrainedModel, PreTrainedTokenizerBase, Trainer, 
    TrainingArguments, DataCollatorWithPadding
)


class GLUESequenceClassificationProfile:
    """Task profile for GLUE-style sequence classification tasks."""
    
    name = "seqcls_glue"
    primary_metric = "accuracy"
    
    def load(self, cfg: Dict[str, Any]) -> Dict[str, Dataset]:
        """Load GLUE dataset from config."""
        task_config = cfg["task"]
        dataset = load_dataset(task_config["dataset"], task_config["subset"])
        
        # Apply sample limits if specified
        if "train" in cfg:
            train_config = cfg["train"]
            if "train_samples" in train_config:
                dataset["train"] = dataset["train"].select(range(min(len(dataset["train"]), train_config["train_samples"])))
            if "eval_samples" in train_config:
                dataset["validation"] = dataset["validation"].select(range(min(len(dataset["validation"]), train_config["eval_samples"])))
        
        return dataset
    
    def tokenize(self, raw_ds: Dict[str, Dataset], tokenizer: PreTrainedTokenizerBase, cfg: Dict[str, Any]) -> Dict[str, Dataset]:
        """Tokenize GLUE dataset with field auto-detection."""
        def preprocess_function(examples):
            """Tokenize examples and preserve labels."""
            # Detect task type based on available fields
            if "question" in examples and "sentence" in examples:
                # QNLI and similar paired tasks
                result = tokenizer(
                    examples["question"], 
                    examples["sentence"],
                    truncation=True, 
                    padding=True, 
                    max_length=128
                )
            elif "sentence" in examples:
                # SST-2 and similar single-text tasks
                result = tokenizer(
                    examples["sentence"], 
                    truncation=True, 
                    padding=True, 
                    max_length=128
                )
            elif "sentence1" in examples and "sentence2" in examples:
                # MNLI, RTE and similar paired sentence tasks
                result = tokenizer(
                    examples["sentence1"],
                    examples["sentence2"],
                    truncation=True,
                    padding=True,
                    max_length=128
                )
            else:
                # Fallback: try to guess the text field
                text_keys = [k for k in examples.keys() if "text" in k.lower() or "sentence" in k.lower()]
                if text_keys:
                    result = tokenizer(
                        examples[text_keys[0]], 
                        truncation=True, 
                        padding=True, 
                        max_length=128
                    )
                else:
                    raise ValueError(f"Could not identify text field(s) in dataset. Available keys: {list(examples.keys())}")
            
            # Make sure labels are preserved
            if "label" in examples:
                result["labels"] = examples["label"]
            return result
        
        return {
            split: dataset.map(preprocess_function, batched=True)
            for split, dataset in raw_ds.items()
        }
    
    def build_trainer(self, model: PreTrainedModel, tokenizer: PreTrainedTokenizerBase,
                     tokenized_ds: Dict[str, Dataset], cfg: Dict[str, Any], callbacks) -> Trainer:
        """Build trainer for sequence classification."""
        train_config = cfg["train"]
        
        # Build training arguments
        training_args = TrainingArguments(
            output_dir="./temp_trainer_output",
            max_steps=train_config.get("max_steps", 500),
            per_device_train_batch_size=train_config.get("per_device_train_batch_size", 8),
            per_device_eval_batch_size=train_config.get("per_device_eval_batch_size", 32),
            learning_rate=train_config.get("lr", 5e-5),
            weight_decay=train_config.get("weight_decay", 0.01),
            warmup_ratio=train_config.get("warmup_ratio", 0.1),
            logging_steps=train_config.get("logging_steps", 50),
            eval_steps=train_config.get("eval_steps", 100),
            save_steps=train_config.get("save_steps", 100),
            eval_strategy="steps",
            save_strategy="no",  # Don't save checkpoints
            load_best_model_at_end=False,
            dataloader_drop_last=False,
            seed=train_config.get("seed", 42),
            report_to=[],  # Disable wandb/tensorboard
        )
        
        return Trainer(
            model=model,
            args=training_args,
            train_dataset=tokenized_ds["train"],
            eval_dataset=tokenized_ds["validation"],
            data_collator=DataCollatorWithPadding(tokenizer),
            callbacks=callbacks or [],
        )
    
    def evaluate(self, model: PreTrainedModel, tokenizer: PreTrainedTokenizerBase,
                tokenized_ds: Dict[str, Dataset], cfg: Dict[str, Any]) -> Dict[str, Any]:
        """Evaluate sequence classification model."""
        train_config = cfg["train"]
        
        # Create minimal trainer for evaluation
        training_args = TrainingArguments(
            output_dir="./temp_eval_output",
            per_device_eval_batch_size=train_config.get("per_device_eval_batch_size", 32),
            seed=train_config.get("seed", 42),
            report_to=[],
        )
        
        trainer = Trainer(
            model=model,
            args=training_args,
            eval_dataset=tokenized_ds["validation"],
            data_collator=DataCollatorWithPadding(tokenizer),
        )
        
        return trainer.evaluate()
    
    def probe_gate(self, probe_eval: Dict[str, Any], cfg: Dict[str, Any]) -> Tuple[bool, Dict[str, Any]]:
        """Check if probe meets accuracy threshold."""
        # Get task-specific threshold
        task_name = cfg["task"]["subset"]
        threshold = self._get_probe_quality_threshold(task_name)
        
        accuracy = probe_eval.get("eval_accuracy") or probe_eval.get("eval_exact_match") or probe_eval.get("accuracy", 0.0)
        passed = accuracy >= threshold
        
        gate_info = {
            "metric": self.primary_metric,
            "value": accuracy,
            "threshold": threshold,
            "passed": passed,
            "task": task_name
        }
        
        return passed, gate_info
    
    def _get_probe_quality_threshold(self, task_name: str) -> float:
        """Get task-specific minimum probe accuracy threshold."""
        task_thresholds = {
            "sst2": 0.75,       # SST-2 sentiment: 75% minimum
            "mnli": 0.70,       # MNLI entailment: 70% minimum  
            "qnli": 0.75,       # QNLI question entailment: 75% minimum
            "qqp": 0.80,        # QQP paraphrase: 80% minimum
            "rte": 0.60,        # RTE small dataset: 60% minimum
            "wnli": 0.55,       # WNLI very small: 55% minimum
            "cola": 0.70,       # CoLA linguistic: 70% minimum
            "mrpc": 0.75,       # MRPC paraphrase: 75% minimum
            "stsb": 0.80,       # STS-B regression converted to classification
        }
        
        return task_thresholds.get(task_name.lower(), 0.70)  # Default 70%