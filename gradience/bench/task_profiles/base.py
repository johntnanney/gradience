"""
Base task profile protocol for Bench.
"""

from typing import Protocol, Dict, Any, Tuple
from datasets import Dataset
from transformers import Trainer, PreTrainedTokenizerBase, PreTrainedModel


class TaskProfile(Protocol):
    """
    Protocol for task-specific implementations in Bench.
    
    Each task profile encapsulates:
    - Dataset loading and preprocessing
    - Model setup and training configuration 
    - Evaluation logic and metrics
    - Probe quality gates
    """
    
    name: str
    primary_metric: str
    primary_metric_key: str
    
    def load(self, cfg: Dict[str, Any]) -> Dict[str, Dataset]:
        """
        Load raw dataset from config.
        
        Args:
            cfg: Full bench configuration
            
        Returns:
            Dictionary with 'train', 'validation' splits
        """
        ...
    
    def tokenize(self, raw_ds: Dict[str, Dataset], tokenizer: PreTrainedTokenizerBase, cfg: Dict[str, Any]) -> Dict[str, Dataset]:
        """
        Tokenize and preprocess dataset.
        
        Args:
            raw_ds: Raw dataset from load()
            tokenizer: Tokenizer to use
            cfg: Full bench configuration
            
        Returns:
            Tokenized dataset ready for training
        """
        ...
    
    def build_trainer(self, model: PreTrainedModel, tokenizer: PreTrainedTokenizerBase, 
                     tokenized_ds: Dict[str, Dataset], cfg: Dict[str, Any], callbacks) -> Trainer:
        """
        Build Trainer instance with task-specific configuration.
        
        Args:
            model: Model to train
            tokenizer: Tokenizer instance
            tokenized_ds: Preprocessed dataset
            cfg: Full bench configuration
            callbacks: Training callbacks (e.g., GradienceCallback)
            
        Returns:
            Configured Trainer instance
        """
        ...
    
    def evaluate(self, model: PreTrainedModel, tokenizer: PreTrainedTokenizerBase,
                tokenized_ds: Dict[str, Dataset], cfg: Dict[str, Any]) -> Dict[str, Any]:
        """
        Evaluate model and return metrics.
        
        Args:
            model: Trained model
            tokenizer: Tokenizer instance
            tokenized_ds: Preprocessed dataset
            cfg: Full bench configuration
            
        Returns:
            Dictionary with evaluation metrics
        """
        ...
    
    def probe_gate(self, probe_eval: Dict[str, Any], cfg: Dict[str, Any]) -> Tuple[bool, Dict[str, Any]]:
        """
        Check if probe meets quality threshold for compression.
        
        Args:
            probe_eval: Results from evaluate()
            cfg: Full bench configuration
            
        Returns:
            (passed: bool, gate_info: dict with details)
        """
        ...