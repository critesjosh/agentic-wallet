"""Pinned first-run configuration for the E2B QLoRA feasibility spike."""

BASE_MODEL_ID = "google/gemma-4-E2B-it"
BASE_MODEL_REVISION = "3e22461f65e89153144f8adb70e3b8c2cc9845a7"
DATASET_VERSION = "wallet-sft-v1"
ERROR_DRIVEN_DATASET_VERSION = "wallet-sft-v2-error-driven"
WORKFLOW_DATASET_VERSION = "wallet-sft-v3-workflow"
PIPELINE_DATASET_VERSION = "wallet-sft-v4-pipeline"
CANDIDATE_PIPELINE_DATASET_VERSION = "wallet-sft-v5-candidate-binding-minimal-route"
TRANSACTION_PIPELINE_DATASET_VERSION = "wallet-sft-v6-transaction-boundary-route"
ACCOUNT_PIPELINE_DATASET_VERSION = "wallet-sft-v7-account-identity-route"
ACCOUNT_DIVERSITY_DATASET_VERSION = "wallet-sft-v8-account-diversity-route"
SUPPORTED_DATASET_VERSIONS = frozenset(
    {
        DATASET_VERSION,
        ERROR_DRIVEN_DATASET_VERSION,
        WORKFLOW_DATASET_VERSION,
        PIPELINE_DATASET_VERSION,
        CANDIDATE_PIPELINE_DATASET_VERSION,
        TRANSACTION_PIPELINE_DATASET_VERSION,
        ACCOUNT_PIPELINE_DATASET_VERSION,
        ACCOUNT_DIVERSITY_DATASET_VERSION,
    }
)
LORA_TARGET_MODULES = ("q_proj", "k_proj", "v_proj", "o_proj")
# Gemma 4's text projections are supported ``nn.Linear`` modules, while its
# vision/audio projections use the unsupported ``Gemma4ClippableLinear``
# wrapper. Target the shared suffixes but exclude those multimodal towers.
LORA_EXCLUDE_PATTERN = r".*(?:vision_tower|audio_tower).*"
