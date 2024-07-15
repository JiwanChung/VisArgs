from PIL import Image
from transformers import InstructBlipProcessor, InstructBlipForConditionalGeneration
import torch

PRETRAINED_PATH = "Salesforce/instructblip-vicuna-7b"
DEVICE = 'cuda'
DTYPE = torch.float16

processor = InstructBlipProcessor.from_pretrained(PRETRAINED_PATH)
model = InstructBlipForConditionalGeneration.from_pretrained(PRETRAINED_PATH, torch_dtype=DTYPE).to(DEVICE)


def instructblip(image_path, prompt):
    image = Image.open(image_path)
    inputs = processor(text=prompt, images=image, return_tensors="pt").to(device=DEVICE, dtype=DTYPE)

    with torch.no_grad():
      generated_ids = model.generate(
          **inputs,
          max_length=len(prompt.split()) + 256,
          do_sample=False
      )
      generated_text = processor.batch_decode(generated_ids, skip_special_tokens=True)[0].strip()
      return generated_text[0]
