from PIL import Image
from transformers import AutoProcessor, LlavaForConditionalGeneration
import torch
import transformers

PRETRAINED_PATH = "llava-hf/llava-1.5-7b-hf"
DEVICE = 'cuda'
MAX_ANSWER_TOKENS = 256
DTYPE = torch.float16

processor = AutoProcessor.from_pretrained(PRETRAINED_PATH)
model = LlavaForConditionalGeneration.from_pretrained(PRETRAINED_PATH, torch_dtype=DTYPE).to(DEVICE)

def llava(image_path, prompt):
  if isinstance(image_path, str):
    image = Image.open(image_path)
  else:
    image = image_path
  inputs = processor(text=prompt, images=image, return_tensors="pt").to(device=DEVICE, dtype=DTYPE)

  with torch.no_grad():
    generated_ids = model.generate(**inputs, max_new_tokens=MAX_ANSWER_TOKENS, do_sample=False)

    generated_text = processor.batch_decode(generated_ids, skip_special_tokens=True, clean_up_tokenization_spaces=False)[0]
    return generated_text.split("ANSWER:")[-1].strip()[0]