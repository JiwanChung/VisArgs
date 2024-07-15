from PIL import Image
from transformers import AutoProcessor, LlavaNextForConditionalGeneration
import transformers
import torch

PRETRAINED_PATH = "llava-hf/llava-v1.6-mistral-7b-hf"
DEVICE = 'cuda'
MAX_ANSWER_TOKENS = 256
DTYPE = torch.float16


processor = AutoProcessor.from_pretrained(PRETRAINED_PATH)
model = LlavaNextForConditionalGeneration.from_pretrained(PRETRAINED_PATH, torch_dtype=DTYPE).to(DEVICE)

def llavanext(image_path, prompt):
    image = Image.open(image_path)
    inputs = processor(text=prompt, images=image, return_tensors="pt").to(device=DEVICE, dtype=DTYPE)

    with torch.no_grad():
      generated_ids = model.generate(**inputs, max_new_tokens=MAX_ANSWER_TOKENS, pad_token_id=2, do_sample=False)

      generated_text = processor.batch_decode(generated_ids, skip_special_tokens=True, clean_up_tokenization_spaces=False)[0]
      return generated_text.split("ANSWER:")[-1].strip()[0]
