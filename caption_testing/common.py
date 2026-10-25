from pathlib import Path

DEFAULT_PROMPT = (
    "Describe this photo using only clearly visible information. "
    "Include the setting, objects, activities, and clothing/accessories "
    "where visible. Keep descriptions attached to the correct person. "
    "Do not guess identities, relationships, medical conditions, emotions, "
    "or specific venues. Omit uncertain details."
)
EXTENSIONS = {".jpg", ".jpeg", ".png", ".webp", ".bmp", ".heic", ".heif", ".tif", ".tiff"}
FIELDS = [
    "batch", "photo", "model", "task", "caption", "seconds",
    "accuracy_1_to_5", "detail_1_to_5", "hallucinations", "error", "prompt",
    "model_key", "status", "device", "dtype", "threads", "max_new_tokens",
    "num_beams", "revision", "resolved_revision", "model_load_seconds",
    "width", "height", "qwen_min_pixels", "qwen_max_pixels",
    "torch_version", "transformers_version", "run_id",
]

def discover_photos(folder, recursive=False, limit=None):
    folder = Path(folder).resolve()
    if not folder.is_dir():
        raise ValueError(f"Photo folder does not exist: {folder}")
    candidates = folder.rglob("*") if recursive else folder.iterdir()
    photos = sorted(
        (p for p in candidates if p.is_file() and p.suffix.lower() in EXTENSIONS),
        key=lambda p: p.relative_to(folder).as_posix().casefold(),
    )
    if limit is not None:
        photos = photos[:limit]
    if not photos:
        raise ValueError("No supported photos found in this folder.")
    return photos

def load_image(path):
    from PIL import Image, ImageOps
    from pillow_heif import register_heif_opener
    register_heif_opener()
    with Image.open(path) as source:
        return ImageOps.exif_transpose(source).convert("RGB")

def synchronize(device):
    if device == "cuda":
        import torch
        torch.cuda.synchronize()

class CaptionAdapter:
    task = "prompted caption"

    def __init__(self, spec, settings):
        self.spec, self.settings = spec, settings
        self.device = settings["device"]
        self.processor = None
        self.model = None

    def load_options(self):
        import torch
        return dict(
            revision=self.spec.revision,
            local_files_only=self.settings["offline"],
            dtype=getattr(torch, self.settings["dtype"]),
            device_map={"": self.device},
        )

    def processor_options(self):
        return dict(revision=self.spec.revision, local_files_only=self.settings["offline"])

    def move_inputs(self, inputs):
        import torch
        # Cast floating image tensors only; token IDs must stay integers.
        return {
            key: value.to(device=self.device, dtype=getattr(torch, self.settings["dtype"]))
            if torch.is_floating_point(value) else value.to(self.device)
            for key, value in inputs.items()
        }

    def generate_options(self):
        return dict(max_new_tokens=self.settings["max_new_tokens"],
                    do_sample=False, num_beams=self.settings["num_beams"])

    def resolved_revision(self):
        return getattr(getattr(self.model, "config", None), "_commit_hash", "") or ""
