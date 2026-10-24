import sys
import torch
from PIL import Image, ImageOps
from transformers import BlipProcessor, BlipForConditionalGeneration

print("Loading model...", flush=True)
model_id = "Salesforce/blip-image-captioning-large"
processor = BlipProcessor.from_pretrained(model_id)
model = BlipForConditionalGeneration.from_pretrained(model_id)
model.eval()

with Image.open(sys.argv[1]) as source:
    image = ImageOps.exif_transpose(source).convert("RGB")

print("Analyzing image...", flush=True)
inputs = processor(images=image, return_tensors="pt")

with torch.inference_mode():
    output = model.generate(**inputs, max_new_tokens=80)

print("\nGenerated caption:")
print(processor.decode(output[0], skip_special_tokens=True))
