from caption_testing.common import CaptionAdapter

class Adapter(CaptionAdapter):
    def __init__(self, spec, settings):
        super().__init__(spec, settings)
        self.task = settings["florence_task"]

    def load(self):
        from transformers import AutoProcessor, Florence2ForConditionalGeneration
        self.processor = AutoProcessor.from_pretrained(self.spec.model_id, **self.processor_options())
        self.model = Florence2ForConditionalGeneration.from_pretrained(
            self.spec.model_id, **self.load_options()).eval()

    def caption(self, image):
        inputs = self.move_inputs(self.processor(text=self.task, images=image, return_tensors="pt"))
        generated = self.model.generate(**inputs, **self.generate_options())
        raw = self.processor.batch_decode(generated, skip_special_tokens=False)[0]
        return self.processor.post_process_generation(raw, task=self.task, image_size=image.size)[self.task]
