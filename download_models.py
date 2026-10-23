"""Download all models used at the hackathon so they work without internet.

Run inside the activated .venv:  python download_models.py
"""
from pathlib import Path

from huggingface_hub import snapshot_download
from ultralytics import YOLO

HF_MODELS = [
    "openai/clip-vit-base-patch32",            # text <-> image search, zero-shot tags
    "Salesforce/blip-image-captioning-large",  # photo captions
    "microsoft/Florence-2-base",               # captions and object tags
    "MCG-NJU/videomae-base-finetuned-kinetics",  # actions in video
    "trpakov/vit-face-expression",             # emotions from face crops
    "sentence-transformers/all-MiniLM-L6-v2",  # text embeddings
]

# Skip formats we do not use (TensorFlow, Flax, ONNX) to save disk space
IGNORE = ["*.h5", "*.msgpack", "*.onnx", "*tflite*", "*.ot", "onnx/*", "openvino/*"]

for name in HF_MODELS:
    snapshot_download(name, ignore_patterns=IGNORE)
    print("OK", name)

weights = Path(__file__).parent / "weights"
weights.mkdir(exist_ok=True)
for w in ["yolo11n.pt", "yolo11s.pt", "yolo11n-pose.pt"]:
    YOLO(str(weights / w))
    print("OK", w)
