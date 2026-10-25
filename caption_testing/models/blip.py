from caption_testing.common import CaptionAdapter

class Adapter(CaptionAdapter):
    task = "caption"

    def load(self):
        from transformers import BlipProcessor, BlipForConditionalGeneration
        self.processor = BlipProcessor.from_pretrained(self.spec.model_id, **self.processor_options())
        self.model = BlipForConditionalGeneration.from_pretrained(
            self.spec.model_id, **self.load_options()).eval()

    def caption(self, image):
        inputs = self.move_inputs(self.processor(images=image, return_tensors="pt"))
        generated = self.model.generate(**inputs, **self.generate_options())
        return self.processor.decode(generated[0], skip_special_tokens=True)
