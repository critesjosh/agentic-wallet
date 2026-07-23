"""Deterministic, leakage-checked data and QLoRA preparation utilities."""

from .data import (
    CoverageDimensions,
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
from .natural_curriculum import (
    NATURAL_CURRICULUM_VERSION,
    load_natural_curriculum,
)
from .pipeline_curriculum import (
    CANDIDATE_PIPELINE_CURRICULUM_VERSION,
    PIPELINE_CURRICULUM_VERSION,
    load_candidate_pipeline_curriculum,
    load_pipeline_curriculum,
)
from .sealed import load_verified_sealed_cases, validate_sealed_commitment
from .blinded import (
    BLINDED_ADAPTER_FILES,
    BLINDED_EVALUATION_CONFIG,
    BLINDED_HASHED_HARNESS_FILES,
    blinded_adapter_sha256,
    blinded_harness_sha256,
    audit_blinded_disjointness,
    load_verified_blinded_cases,
    sha256_named_files,
    validate_blinded_commitment,
)
from .formatting import (
    completion_json,
    instruction_prompt,
    render_training_pair,
    tokenize_completion_only,
)
from .evaluation import (
    balanced_semantic_subset,
    DevelopmentCaseResult,
    DevelopmentReport,
    evaluate_development_examples,
)

__all__ = [
    "BLINDED_ADAPTER_FILES",
    "BLINDED_EVALUATION_CONFIG",
    "BLINDED_HASHED_HARNESS_FILES",
    "CoverageDimensions",
    "DatasetValidationReport",
    "DevelopmentCaseResult",
    "DevelopmentReport",
    "balanced_semantic_subset",
    "ERROR_DRIVEN_GENERATOR_VERSION",
    "GENERATOR_VERSION",
    "NATURAL_CURRICULUM_VERSION",
    "PIPELINE_CURRICULUM_VERSION",
    "CANDIDATE_PIPELINE_CURRICULUM_VERSION",
    "TrainingExample",
    "audit_blinded_disjointness",
    "blinded_adapter_sha256",
    "blinded_harness_sha256",
    "completion_json",
    "generate_error_driven_training_examples",
    "generate_training_examples",
    "evaluate_development_examples",
    "instruction_prompt",
    "load_training_examples",
    "load_natural_curriculum",
    "load_pipeline_curriculum",
    "load_candidate_pipeline_curriculum",
    "load_verified_sealed_cases",
    "load_verified_blinded_cases",
    "render_training_pair",
    "tokenize_completion_only",
    "validate_training_dataset",
    "validate_sealed_commitment",
    "validate_blinded_commitment",
    "sha256_named_files",
]
