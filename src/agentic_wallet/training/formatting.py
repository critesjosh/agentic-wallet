"""Prompt/completion rendering and completion-only token masking for SFT."""

from __future__ import annotations

import json
from typing import Any

from ..tool_contract import (
    dialogue_route_prompt,
    legacy_dialogue_route_prompt,
    dialogue_turn_prompt,
    tool_call_prompt,
)
from .data import TrainingExample


def completion_json(example: TrainingExample) -> str:
    return json.dumps(example.target, sort_keys=True, separators=(",", ":"))


def instruction_prompt(example: TrainingExample) -> str:
    if example.kind == "tool_call":
        return tool_call_prompt(example.context, example.available_actions)
    if example.kind == "dialogue_route":
        renderer = (
            dialogue_route_prompt
            if set(example.target) == {"proposed_action"}
            else legacy_dialogue_route_prompt
        )
        return renderer(
            example.context,
            example.available_actions,
            example.suggested_action_ids,
        )
    return dialogue_turn_prompt(
        example.context,
        example.available_actions,
        example.suggested_action_ids,
    )


def render_training_pair(processor: Any, example: TrainingExample) -> tuple[str, str]:
    """Render the same user request shape as inference plus its target completion."""

    prompt = instruction_prompt(example)
    target = completion_json(example)
    if not getattr(processor, "chat_template", None):
        return prompt, prompt + target

    user = {"role": "user", "content": [{"type": "text", "text": prompt}]}
    assistant = {
        "role": "assistant",
        "content": [{"type": "text", "text": target}],
    }
    prompt_text = processor.apply_chat_template(
        [user], tokenize=False, add_generation_prompt=True
    )
    full_text = processor.apply_chat_template(
        [user, assistant], tokenize=False, add_generation_prompt=False
    )
    return prompt_text, full_text


def tokenize_completion_only(
    processor: Any, example: TrainingExample, *, max_length: int
) -> dict[str, list[int]]:
    """Tokenize and mask every prompt token to ``-100``."""

    prompt_text, full_text = render_training_pair(processor, example)
    tokenizer = getattr(processor, "tokenizer", processor)
    prompt_tokens = tokenizer(
        prompt_text,
        add_special_tokens=False,
        truncation=True,
        max_length=max_length,
    )
    full_tokens = tokenizer(
        full_text,
        add_special_tokens=False,
        truncation=True,
        max_length=max_length,
    )
    prompt_ids = list(prompt_tokens["input_ids"])
    input_ids = list(full_tokens["input_ids"])
    if input_ids[: len(prompt_ids)] != prompt_ids:
        raise ValueError("chat template does not preserve the inference prompt prefix")
    if len(input_ids) <= len(prompt_ids):
        raise ValueError(f"completion was fully truncated for {example.id}")
    record = {key: list(value) for key, value in full_tokens.items()}
    record["labels"] = [-100] * len(prompt_ids) + input_ids[len(prompt_ids) :]
    return record
