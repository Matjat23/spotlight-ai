from dataclasses import dataclass
from importlib import import_module

@dataclass(frozen=True)
class ModelSpec:
    model_id: str
    module: str
    default_tokens: int
    beams: int = 1
    revision: str = "main"

MODELS = {
    "blip": ModelSpec("Salesforce/blip-image-captioning-large", "blip", 80, 3),
    "florence-base": ModelSpec("florence-community/Florence-2-base", "florence", 256, 3),
    "florence-large": ModelSpec("florence-community/Florence-2-large", "florence", 256, 3),
    "qwen-vl": ModelSpec("Qwen/Qwen2.5-VL-3B-Instruct", "qwen", 256),
    "smolvlm-500m": ModelSpec("HuggingFaceTB/SmolVLM-500M-Instruct", "smolvlm", 256),
    "smolvlm2-2.2b": ModelSpec("HuggingFaceTB/SmolVLM2-2.2B-Instruct", "smolvlm", 256),
    "moondream2": ModelSpec("vikhyatk/moondream2", "moondream", 256, revision="2025-06-21"),
}

def adapter_class(spec):
    return import_module("caption_testing.models." + spec.module).Adapter
