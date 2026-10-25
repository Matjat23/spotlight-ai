from caption_testing.common import CaptionAdapter

class Adapter(CaptionAdapter):
    task = "visual query caption"

    def load(self):
        from transformers import AutoModelForCausalLM
        # The official Moondream implementation uses custom Python code.
        self.model = AutoModelForCausalLM.from_pretrained(
            self.spec.model_id, trust_remote_code=True, **self.load_options()).eval()

    def caption(self, image):
        return self.model.query(image, self.settings["prompt"], settings={
            "max_tokens": self.settings["max_new_tokens"], "temperature": 0,
        })["answer"]
