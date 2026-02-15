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


# ---------------------------------------------------------------------------
# MNLI prompts
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
) -> str:
    """Format an MNLI example as a prompted NLI task.

    Parameters
    ----------
    premise : the premise text
    hypothesis : the hypothesis text
    few_shot : if True, include 3-shot examples (recommended)

    Returns
    -------
    Formatted prompt string ready for causal LM generation.
    """
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
) -> str:
    """Format a QNLI example as a prompted entailment task.

    Parameters
    ----------
    question : the question text
    sentence : the candidate answer sentence
    few_shot : if True, include 2-shot examples (recommended)

    Returns
    -------
    Formatted prompt string ready for causal LM generation.
    """
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
