"""Local Hugging Face Transformers inference for Gemma 4.

The heavyweight ML dependencies are imported only by :meth:`load`, keeping the
core package usable without the optional ``ml`` dependencies. This provider has
no native grammar support: output is post-hoc validated and therefore must meet
the measured structured-validity gate before it can be selected for release.
"""

from __future__ import annotations

import json
from typing import Any

from ..inference import InferenceError, InferenceProvider
from ..schemas.tool_call import ToolCall
from ..schemas.dialogue import ModelDialogueTurn
from ..tool_contract import (
    dialogue_turn_prompt,
    tool_call_prompt,
    validate_dialogue_turn,
)


def _extract_json(text: str) -> dict[str, Any]:
    """Require the entire generated output to be exactly one JSON object."""

    try:
        value = json.loads(text.strip())
    except json.JSONDecodeError as exc:
        raise InferenceError("model output was not exactly one JSON object") from exc
    if not isinstance(value, dict):
        raise InferenceError("model output must be a JSON object")
    return value


class LocalTransformersProvider(InferenceProvider):
    """Run the target Gemma model locally with deterministic 4-bit inference."""

    name = "local-transformers"

    def __init__(
        self,
        model_id: str = "google/gemma-4-E2B-it",
        revision: str | None = None,
        adapter_path: str | None = None,
        load_in_4bit: bool = True,
        max_new_tokens: int = 256,
        device: str = "cuda",
    ) -> None:
        self.model_id = model_id
        self.revision = revision
        self.adapter_path = adapter_path
        self.load_in_4bit = load_in_4bit
        self.max_new_tokens = max_new_tokens
        self.device = device
        self._processor: Any | None = None
        self._model: Any | None = None

    def load(self) -> None:
        """Load the processor and model once, importing optional dependencies lazily."""

        if self._model is not None:
            return

        try:
            import torch
            from peft import PeftModel
            from transformers import (
                AutoModelForMultimodalLM,
                AutoProcessor,
                BitsAndBytesConfig,
            )
        except ImportError as exc:
            raise InferenceError(
                'local inference requires the optional dependencies: pip install -e ".[ml]"'
            ) from exc

        if self.device.startswith("cuda") and not torch.cuda.is_available():
            raise InferenceError("CUDA was requested but is not available")

        quantization_config = None
        if self.load_in_4bit:
            quantization_config = BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_quant_type="nf4",
                bnb_4bit_compute_dtype=torch.bfloat16,
                bnb_4bit_use_double_quant=True,
            )

        self._processor = AutoProcessor.from_pretrained(
            self.model_id, revision=self.revision
        )
        self._model = AutoModelForMultimodalLM.from_pretrained(
            self.model_id,
            revision=self.revision,
            device_map=self.device,
            quantization_config=quantization_config,
            dtype=torch.bfloat16,
        )
        if self.adapter_path is not None:
            self._model = PeftModel.from_pretrained(
                self._model, self.adapter_path, is_trainable=False
            )
        self._model.eval()

    def _build_prompt(self, context: dict, available_actions: list[str]) -> str:
        request_text = tool_call_prompt(context, available_actions)
        return self._render_prompt(request_text)

    def _render_prompt(self, request_text: str) -> str:
        messages = [
            {"role": "user", "content": [{"type": "text", "text": request_text}]},
        ]

        # Loading is deferred until generation, so prompt construction remains
        # lightweight and independently testable.
        if self._processor is None:
            self.load()
        if getattr(self._processor, "chat_template", None):
            return self._processor.apply_chat_template(
                messages, tokenize=False, add_generation_prompt=True
            )
        return request_text

    def _generate(self, prompt: str) -> str:
        if self._model is None or self._processor is None:
            self.load()

        inputs = self._processor(
            text=prompt, return_tensors="pt", add_special_tokens=False
        )
        inputs = {name: value.to(self.device) for name, value in inputs.items()}
        input_length = inputs["input_ids"].shape[-1]
        generated = self._model.generate(
            **inputs,
            do_sample=False,
            max_new_tokens=self.max_new_tokens,
        )
        new_tokens = generated[0, input_length:]
        return self._processor.decode(new_tokens, skip_special_tokens=True)

    def _build_dialogue_prompt(
        self,
        context: dict,
        available_actions: list[str],
        suggested_action_ids: list[str],
    ) -> str:
        return self._render_prompt(
            dialogue_turn_prompt(
                context, available_actions, suggested_action_ids
            )
        )

    def propose_tool_call(
        self, context: dict, available_actions: list[str]
    ) -> ToolCall:
        prompt = self._build_prompt(context, available_actions)
        raw = _extract_json(self._generate(prompt))
        return self._validate(raw, available_actions)

    def propose_dialogue_turn(
        self,
        context: dict,
        available_actions: list[str],
        suggested_action_ids: list[str],
    ) -> ModelDialogueTurn:
        prompt = self._build_dialogue_prompt(
            context, available_actions, suggested_action_ids
        )
        raw = _extract_json(self._generate(prompt))
        return validate_dialogue_turn(
            raw, available_actions, suggested_action_ids
        )
