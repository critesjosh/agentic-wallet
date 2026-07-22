from __future__ import annotations

import re

from agentic_wallet.training import (
    load_pipeline_curriculum,
    generate_training_examples,
    render_training_pair,
    tokenize_completion_only,
)
from pathlib import Path
from agentic_wallet.training.config import (
    LORA_EXCLUDE_PATTERN,
    LORA_TARGET_MODULES,
)


class FakeTokenizer:
    pad_token_id = 0

    def __call__(self, text, **_kwargs):
        return {
            "input_ids": [ord(char) for char in text],
            "attention_mask": [1] * len(text),
        }


class FakeProcessor:
    chat_template = "fake"

    def __init__(self):
        self.tokenizer = FakeTokenizer()

    def apply_chat_template(self, messages, *, add_generation_prompt, **_kwargs):
        rendered = "<user>" + messages[0]["content"][0]["text"] + "</user>"
        if len(messages) == 2:
            rendered += "<assistant>" + messages[1]["content"][0]["text"] + "</assistant>"
        elif add_generation_prompt:
            rendered += "<assistant>"
        return rendered


def test_training_pair_preserves_inference_prompt_prefix():
    example = generate_training_examples(tool_count=1, dialogue_count=1)[0]
    prompt, full = render_training_pair(FakeProcessor(), example)
    assert full.startswith(prompt)
    assert example.context["user_request"] in prompt


def test_completion_only_mask_hides_every_prompt_token():
    example = generate_training_examples(tool_count=1, dialogue_count=1)[0]
    record = tokenize_completion_only(FakeProcessor(), example, max_length=10_000)
    first_supervised = record["labels"].index(next(x for x in record["labels"] if x != -100))
    prompt, _ = render_training_pair(FakeProcessor(), example)
    assert first_supervised == len(prompt)
    assert all(value == -100 for value in record["labels"][:first_supervised])
    assert any(value != -100 for value in record["labels"][first_supervised:])


def test_pipeline_route_and_repair_render_exact_runtime_phases():
    source = Path(__file__).resolve().parents[1] / "data" / "training" / "natural_v3_source.jsonl"
    examples = load_pipeline_curriculum(source)
    route = next(item for item in examples if item.kind == "dialogue_route")
    repair = next(
        item
        for item in examples
        if item.context.get("phase") == "repair_tool_arguments"
    )
    route_prompt, _ = render_training_pair(FakeProcessor(), route)
    repair_prompt, _ = render_training_pair(FakeProcessor(), repair)
    assert "route_dialogue" in route_prompt
    assert "repair_tool_arguments" in repair_prompt
    assert "previous_output" in repair_prompt


def test_lora_targets_supported_inner_projections_and_excludes_towers():
    q_proj = "model.language_model.layers.0.self_attn.q_proj"
    o_proj = "model.language_model.layers.4.self_attn.o_proj"
    unrelated = "model.language_model.layers.0.mlp.down_proj"
    vision = "model.vision_tower.encoder.layers.0.self_attn.q_proj"

    assert any(q_proj.endswith(f".{target}") for target in LORA_TARGET_MODULES)
    assert any(o_proj.endswith(f".{target}") for target in LORA_TARGET_MODULES)
    assert not any(unrelated.endswith(f".{target}") for target in LORA_TARGET_MODULES)
    assert re.fullmatch(LORA_EXCLUDE_PATTERN, vision)
