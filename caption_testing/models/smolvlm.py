from caption_testing.common import CaptionAdapter

class Adapter(CaptionAdapter):
    def load(self):
        from transformers import AutoProcessor, AutoModelForImageTextToText
        self.processor = AutoProcessor.from_pretrained(self.spec.model_id, **self.processor_options())
        self.model = AutoModelForImageTextToText.from_pretrained(
            self.spec.model_id, attn_implementation="sdpa", **self.load_options()).eval()

    def caption(self, image):
        messages = [{"role": "user", "content": [
            {"type": "image"}, {"type": "text", "text": self.settings["prompt"]}]}]
        text = self.processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        inputs = self.move_inputs(self.processor(text=text, images=[image], return_tensors="pt"))
        generated = self.model.generate(**inputs, **self.generate_options())
        answer = generated[:, inputs["input_ids"].shape[1]:]
        return self.processor.batch_decode(answer, skip_special_tokens=True)[0]
