"""
GSM8K causal language modeling task profile.
"""

import re
from typing import Dict, Any, Tuple

from datasets import Dataset, load_dataset
from transformers import (
    PreTrainedModel, PreTrainedTokenizerBase, Trainer, 
    TrainingArguments, DataCollatorForLanguageModeling, 
    AutoModelForCausalLM
)
import torch


class GSM8KCausalLMProfile:
    """Task profile for GSM8K math reasoning with causal language modeling."""
    
    name = "gsm8k_causal_lm"
    primary_metric = "exact_match"
    
    def load(self, cfg: Dict[str, Any]) -> Dict[str, Dataset]:
        """Load GSM8K dataset."""
        # GSM8K has train/test splits, we'll use test as validation
        dataset = load_dataset("gsm8k", "main")
        
        # Apply sample limits if specified
        task_config = cfg.get("task", {})
        eval_max_samples = task_config.get("eval_max_samples", 500)
        
        if "train" in cfg:
            train_config = cfg["train"]
            if "train_samples" in train_config:
                dataset["train"] = dataset["train"].select(range(min(len(dataset["train"]), train_config["train_samples"])))
        
        # Use test set as validation and apply eval limits
        dataset["validation"] = dataset["test"].select(range(min(len(dataset["test"]), eval_max_samples)))
        
        return dataset
    
    def tokenize(self, raw_ds: Dict[str, Dataset], tokenizer: PreTrainedTokenizerBase, cfg: Dict[str, Any]) -> Dict[str, Dataset]:
        """Tokenize GSM8K for response-only loss training."""
        # Ensure tokenizer has pad token for causal LM
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token
            
        def preprocess_function(examples):
            """Format as instruction-response pairs with response-only loss."""
            prompts = []
            completions = []
            
            for question, answer in zip(examples["question"], examples["answer"]):
                # Format as instruction-response
                prompt = f"Question: {question}\nAnswer:"
                completion = f" {answer}"
                
                prompts.append(prompt)
                completions.append(completion)
            
            # Tokenize prompts and completions separately
            prompt_tokens = tokenizer(
                prompts, 
                truncation=True, 
                padding=False,
                max_length=256,  # Reserve space for completion
                add_special_tokens=True
            )
            
            completion_tokens = tokenizer(
                completions,
                truncation=True,
                padding=False, 
                max_length=128,
                add_special_tokens=False  # Don't add special tokens to completion
            )
            
            # Combine and create labels for response-only loss
            input_ids = []
            attention_mask = []
            labels = []
            
            for prompt_ids, prompt_mask, comp_ids, comp_mask in zip(
                prompt_tokens["input_ids"], prompt_tokens["attention_mask"],
                completion_tokens["input_ids"], completion_tokens["attention_mask"]
            ):
                # Combine prompt + completion
                combined_ids = prompt_ids + comp_ids
                combined_mask = prompt_mask + comp_mask
                
                # Create labels: -100 for prompt (ignored), actual tokens for completion
                combined_labels = [-100] * len(prompt_ids) + comp_ids
                
                input_ids.append(combined_ids)
                attention_mask.append(combined_mask)
                labels.append(combined_labels)
            
            return {
                "input_ids": input_ids,
                "attention_mask": attention_mask,
                "labels": labels
            }
        
        # Only tokenize train set for training; keep raw validation for evaluation
        result = {}
        if "train" in raw_ds:
            result["train"] = raw_ds["train"].map(preprocess_function, batched=True, remove_columns=raw_ds["train"].column_names)
        if "validation" in raw_ds:
            result["validation"] = raw_ds["validation"]  # Keep raw for generation evaluation
            
        return result
    
    def build_trainer(self, model: PreTrainedModel, tokenizer: PreTrainedTokenizerBase,
                     tokenized_ds: Dict[str, Dataset], cfg: Dict[str, Any], callbacks) -> Trainer:
        """Build trainer for causal language modeling."""
        train_config = cfg["train"]
        
        # Build training arguments
        training_args = TrainingArguments(
            output_dir="./temp_trainer_output",
            max_steps=train_config.get("max_steps", 1500),
            per_device_train_batch_size=train_config.get("per_device_train_batch_size", 1),
            gradient_accumulation_steps=train_config.get("gradient_accumulation_steps", 16),
            learning_rate=train_config.get("learning_rate", 1e-4),
            weight_decay=train_config.get("weight_decay", 0.0),
            warmup_ratio=train_config.get("warmup_ratio", 0.03),
            logging_steps=train_config.get("logging_steps", 10),
            eval_steps=train_config.get("eval_steps", 500),
            save_steps=train_config.get("save_steps", 500),
            eval_strategy="no",  # Disable in-training eval to prevent crashes
            do_eval=False,  # Explicitly disable evaluation
            save_strategy=train_config.get("save_strategy", "no"),
            load_best_model_at_end=False,
            dataloader_drop_last=False,
            seed=train_config.get("seed", 42),
            report_to=[],  # Disable wandb/tensorboard
            bf16=True if train_config.get("torch_dtype") == "bf16" else False,
            remove_unused_columns=False,  # Prevent column removal issues
        )
        
        return Trainer(
            model=model,
            args=training_args,
            train_dataset=tokenized_ds["train"],
            eval_dataset=None,  # Disable eval dataset to prevent crashes
            data_collator=DataCollatorForLanguageModeling(tokenizer, mlm=False),
            callbacks=callbacks or [],
        )
    
    def evaluate(self, model: PreTrainedModel, tokenizer: PreTrainedTokenizerBase,
                tokenized_ds: Dict[str, Dataset], cfg: Dict[str, Any]) -> Dict[str, Any]:
        """Evaluate model using generation and exact match."""
        # Use raw validation dataset for generation
        val_dataset = tokenized_ds["validation"]
        
        # Generation config from task config
        generation_config = cfg["task"].get("generation", {})
        max_new_tokens = generation_config.get("max_new_tokens", 128)
        do_sample = generation_config.get("do_sample", False)
        temperature = generation_config.get("temperature", 0.0)
        num_beams = generation_config.get("num_beams", 1)
        
        correct = 0
        total = 0
        
        model.eval()
        device = next(model.parameters()).device
        
        for example in val_dataset:
            question = example["question"]
            gold_answer = example["answer"]
            
            # Format prompt
            prompt = f"Question: {question}\nAnswer:"
            
            # Tokenize input
            inputs = tokenizer(prompt, return_tensors="pt", truncation=True, max_length=256)
            inputs = {k: v.to(device) for k, v in inputs.items()}
            
            # Generate response
            with torch.no_grad():
                outputs = model.generate(
                    **inputs,
                    max_new_tokens=max_new_tokens,
                    do_sample=do_sample,
                    temperature=temperature if do_sample else None,
                    num_beams=num_beams,
                    pad_token_id=tokenizer.eos_token_id,
                )
            
            # Decode generated response (skip input tokens)
            generated_tokens = outputs[0][inputs["input_ids"].shape[1]:]
            generated_text = tokenizer.decode(generated_tokens, skip_special_tokens=True)
            
            # Extract numerical answer
            pred_answer = self._extract_answer(generated_text)
            gold_answer_num = self._extract_answer(gold_answer)
            
            if pred_answer == gold_answer_num:
                correct += 1
            total += 1
        
        accuracy = correct / total if total > 0 else 0.0
        
        return {
            "eval_exact_match": accuracy,
            "eval_accuracy": accuracy,  # Alias for compatibility
            "eval_correct": correct,
            "eval_total": total,
            "eval_samples": total,
        }
    
    def probe_gate(self, probe_eval: Dict[str, Any], cfg: Dict[str, Any]) -> Tuple[bool, Dict[str, Any]]:
        """Check if probe meets exact match threshold."""
        # Use probe gate config or default threshold
        probe_gate_config = cfg["task"].get("probe_gate", {})
        threshold = probe_gate_config.get("min_value", 0.15)
        
        exact_match = probe_eval["eval_exact_match"]
        passed = exact_match >= threshold
        
        gate_info = {
            "metric": self.primary_metric,
            "value": exact_match,
            "threshold": threshold,
            "passed": passed,
            "task": "gsm8k"
        }
        
        return passed, gate_info
    
    def _extract_answer(self, text: str) -> str:
        """Extract numerical answer from GSM8K response."""
        # Look for patterns like "#### 42" or final number in text
        # GSM8K answers typically end with "#### [number]"
        
        # Try to find #### pattern first
        match = re.search(r'####\s*([0-9,]+(?:\.[0-9]+)?)', text)
        if match:
            return match.group(1).replace(',', '')
        
        # Fallback: find last number in text
        numbers = re.findall(r'[0-9,]+(?:\.[0-9]+)?', text)
        if numbers:
            return numbers[-1].replace(',', '')
        
        return ""