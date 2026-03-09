#!/usr/bin/env python3
"""
Per-task training data formatting for M1 experiment.

Each task has a `format_<task>` function that takes a dataset example
and returns a formatted string for causal LM fine-tuning.

Supported tasks: sql, chat, math, code
"""

from __future__ import annotations


def format_sql(example: dict) -> str:
    """Format SQL generation example: schema + question -> SQL query.

    Dataset: b-mc2/sql-create-context
    Fields: context (schema), question, answer (SQL)
    """
    return f"### Schema:\n{example['context']}\n\n### Question:\n{example['question']}\n\n### SQL:\n{example['answer']}"


def format_chat(example: dict) -> str:
    """Format instruction-following example in Alpaca format.

    Dataset: yahma/alpaca-cleaned
    Fields: instruction, input (optional), output
    """
    instruction = example["instruction"]
    inp = example.get("input", "")
    output = example["output"]

    if inp and inp.strip():
        return f"### Instruction:\n{instruction}\n\n### Input:\n{inp}\n\n### Response:\n{output}"
    return f"### Instruction:\n{instruction}\n\n### Response:\n{output}"


def format_math(example: dict) -> str:
    """Format GSM8K chain-of-thought math example.

    Dataset: gsm8k (main)
    Fields: question, answer (includes reasoning + #### final_answer)
    """
    return f"### Question:\n{example['question']}\n\n### Answer:\n{example['answer']}"


def format_code(example: dict) -> str:
    """Format code generation example from docstring.

    Dataset: sahil2801/CodeAlpaca-20k
    Fields: instruction, input (optional), output
    """
    instruction = example["instruction"]
    inp = example.get("input", "")
    output = example["output"]

    if inp and inp.strip():
        return f"### Instruction:\n{instruction}\n\n### Input:\n{inp}\n\n### Code:\n{output}"
    return f"### Instruction:\n{instruction}\n\n### Code:\n{output}"


# Registry mapping task name -> formatter
TASK_FORMATTERS = {
    "sql": format_sql,
    "chat": format_chat,
    "math": format_math,
    "code": format_code,
}


def get_formatter(task_name: str):
    """Get the formatting function for a task.

    Raises ValueError if task_name is not recognized.
    """
    fn = TASK_FORMATTERS.get(task_name)
    if fn is None:
        raise ValueError(f"Unknown task '{task_name}'. Available: {sorted(TASK_FORMATTERS.keys())}")
    return fn
