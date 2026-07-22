"""Deterministic, leakage-checked data and QLoRA preparation utilities."""

from .data import (
    DatasetValidationReport,
    TrainingExample,
    load_training_examples,
    validate_training_dataset,
)
from .generator import (
    ERROR_DRIVEN_GENERATOR_VERSION,
    GENERATOR_VERSION,
    generate_error_driven_training_examples,
    generate_training_examples,
)
from .formatting import (
    completion_json,
    instruction_prompt,
    render_training_pair,
    tokenize_completion_only,
)

__all__ = [
    "DatasetValidationReport",
    "ERROR_DRIVEN_GENERATOR_VERSION",
    "GENERATOR_VERSION",
    "TrainingExample",
    "completion_json",
    "generate_error_driven_training_examples",
    "generate_training_examples",
    "instruction_prompt",
    "load_training_examples",
    "render_training_pair",
    "tokenize_completion_only",
    "validate_training_dataset",
]
