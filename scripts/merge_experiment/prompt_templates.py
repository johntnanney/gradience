"""
Prompt templates for NLI evaluation via causal LM generation.

Formats MNLI and QNLI examples into few-shot prompts suitable for
Mistral-7B or similar causal language models.  Parses generated text
to extract predicted labels.

Usage::

    from scripts.merge_experiment.prompt_templates import (
        format_mnli_prompt,
        format_qnli_prompt,
        parse_nli_response,
    )

    prompt = format_mnli_prompt(premise, hypothesis)
    # Feed to model, get response text
    label = parse_nli_response(response)  # "entailment" | "neutral" | "contradiction"
"""

from __future__ import annotations

import re
from typing import Literal

TemplateMode = Literal["predibase", "custom"]


# ---------------------------------------------------------------------------
# Predibase templates (numeric output — from lora_bakeoff metadata.yaml)
# ---------------------------------------------------------------------------

_PREDIBASE_MNLI_TEMPLATE = (
    "You are given a premise and a hypothesis below. If the premise entails the "
    "hypothesis, return 0. If the premise contradicts the hypothesis, return 2. "
    "Otherwise, if the premise does neither, return 1.\n\n"
    "### Premise: {premise}\n\n"
    "### Hypothesis: {hypothesis}\n\n"
    "### Label: "
)

_PREDIBASE_QNLI_TEMPLATE = (
    "You are provided a question and a corresponding response below. If the "
    "response properly answers the question, please return 0. Otherwise, please "
    "return 1.\n\n"
    "### Question: {question}\n\n"
    "### Response: {sentence}\n\n"
    "### Label: "
)

# Numeric → word label maps for Predibase output parsing
_PREDIBASE_MNLI_MAP = {0: "entailment", 1: "neutral", 2: "contradiction"}
_PREDIBASE_QNLI_MAP = {0: "entailment", 1: "not_entailment"}


# ---------------------------------------------------------------------------
# Custom templates (word-label output — for self-trained adapters)
# ---------------------------------------------------------------------------

_MNLI_TEMPLATE = """Classify the relationship between the premise and hypothesis.
Answer with exactly one word: entailment, neutral, or contradiction.

Premise: {premise}
Hypothesis: {hypothesis}
Answer:"""

_MNLI_FEW_SHOT = """Classify the relationship between the premise and hypothesis.
Answer with exactly one word: entailment, neutral, or contradiction.

Premise: The cat sat on the mat.
Hypothesis: An animal was on the mat.
Answer: entailment

Premise: The restaurant was crowded on Friday evening.
Hypothesis: The restaurant was empty on Friday evening.
Answer: contradiction

Premise: She drove to work today.
Hypothesis: She usually takes the bus.
Answer: neutral

Premise: {premise}
Hypothesis: {hypothesis}
Answer:"""


def format_mnli_prompt(
    premise: str,
    hypothesis: str,
    few_shot: bool = True,
    template_mode: TemplateMode = "custom",
) -> str:
    """Format an MNLI example as a prompted NLI task.

    Parameters
    ----------
    premise : the premise text
    hypothesis : the hypothesis text
    few_shot : if True, include 3-shot examples (ignored for predibase mode)
    template_mode : "predibase" for numeric output, "custom" for word labels

    Returns
    -------
    Formatted prompt string ready for causal LM generation.
    """
    if template_mode == "predibase":
        return _PREDIBASE_MNLI_TEMPLATE.format(
            premise=premise.strip(), hypothesis=hypothesis.strip()
        )
    template = _MNLI_FEW_SHOT if few_shot else _MNLI_TEMPLATE
    return template.format(premise=premise.strip(), hypothesis=hypothesis.strip())


# ---------------------------------------------------------------------------
# QNLI prompts
# ---------------------------------------------------------------------------

_QNLI_TEMPLATE = """Does the sentence answer the question?
Answer with exactly one word: entailment or not_entailment.

Question: {question}
Sentence: {sentence}
Answer:"""

_QNLI_FEW_SHOT = """Does the sentence answer the question?
Answer with exactly one word: entailment or not_entailment.

Question: What is the capital of France?
Sentence: Paris is the capital and most populous city of France.
Answer: entailment

Question: When was the Declaration of Independence signed?
Sentence: The document was drafted by Thomas Jefferson.
Answer: not_entailment

Question: {question}
Sentence: {sentence}
Answer:"""


def format_qnli_prompt(
    question: str,
    sentence: str,
    few_shot: bool = True,
    template_mode: TemplateMode = "custom",
) -> str:
    """Format a QNLI example as a prompted entailment task.

    Parameters
    ----------
    question : the question text
    sentence : the candidate answer sentence
    few_shot : if True, include 2-shot examples (ignored for predibase mode)
    template_mode : "predibase" for numeric output, "custom" for word labels

    Returns
    -------
    Formatted prompt string ready for causal LM generation.
    """
    if template_mode == "predibase":
        return _PREDIBASE_QNLI_TEMPLATE.format(
            question=question.strip(), sentence=sentence.strip()
        )
    template = _QNLI_FEW_SHOT if few_shot else _QNLI_TEMPLATE
    return template.format(question=question.strip(), sentence=sentence.strip())


# ---------------------------------------------------------------------------
# Response parsing
# ---------------------------------------------------------------------------

# Canonical labels for each task
MNLI_LABELS = {"entailment", "neutral", "contradiction"}
QNLI_LABELS = {"entailment", "not_entailment"}
ALL_LABELS = MNLI_LABELS | QNLI_LABELS


def parse_nli_response(response: str) -> str:
    """Extract an NLI label from generated text.

    Performs case-insensitive matching against known labels.
    Returns the first recognized label found in the response,
    or "unknown" if no label is detected.

    Parameters
    ----------
    response : raw generated text from the model

    Returns
    -------
    Lowercase label string: one of "entailment", "neutral",
    "contradiction", "not_entailment", or "unknown".
    """
    text = response.strip().lower()

    # Check for exact match on first word
    first_word = re.split(r"[\s.,;:!?]", text)[0] if text else ""
    if first_word in ALL_LABELS:
        return first_word

    # Check for "not_entailment" before "entailment" (substring match order matters)
    if "not_entailment" in text:
        return "not_entailment"
    if "contradiction" in text:
        return "contradiction"
    if "neutral" in text:
        return "neutral"
    if "entailment" in text:
        return "entailment"

    return "unknown"


def parse_nli_response_numeric(response: str, task: str) -> str:
    """Extract an NLI label from numeric output (Predibase-style adapters).

    Predibase adapters output digits (0/1/2) instead of word labels.
    This parser extracts the first digit and maps it to a canonical label.

    Parameters
    ----------
    response : raw generated text from the model
    task : "mnli" or "qnli" — determines the digit→label mapping

    Returns
    -------
    Canonical lowercase label string, or "unknown" if no valid digit found.
    """
    label_map = _PREDIBASE_MNLI_MAP if task == "mnli" else _PREDIBASE_QNLI_MAP
    text = response.strip()

    # Extract first digit character
    for ch in text:
        if ch.isdigit():
            num = int(ch)
            if num in label_map:
                return label_map[num]
            break  # found a digit but it's out of range

    return "unknown"
