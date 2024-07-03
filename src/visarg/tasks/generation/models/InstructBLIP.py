from PIL import Image
from transformers import InstructBlipProcessor, InstructBlipForConditionalGeneration
import torch

PRETRAINED_PATH = "Salesforce/instructblip-vicuna-7b"
PROMPT = "Your task is to answer what the image wants to say. You should answer in only one sentence without an unnecessary prefix. ANSWER:"
DEVICE = 'cuda'
MAX_ANSWER_TOKENS = 256
DTYPE = torch.float16
FP16 = True

processor = InstructBlipProcessor.from_pretrained(PRETRAINED_PATH)
model = InstructBlipForConditionalGeneration.from_pretrained(PRETRAINED_PATH, load_in_8bit=FP16, torch_dtype=DTYPE)

def prompt(prefix : str = None, postfix : str = None, need_base : bool = True):
    if need_base:
        text = PROMPT
    else:
        text = ""
    
    if prefix is not None:
        text = prefix + text
        
    if postfix is not None:
        text = text + postfix
    
    return text

def instructblip(image_path, prompt=PROMPT):
    image = Image.open(image_path)
    inputs = processor(text=prompt, images=image, return_tensors="pt").to(device=DEVICE, dtype=DTYPE)

    generated_ids = model.generate(
        **inputs,
        # do_sample=False,
        num_beams=5,
        max_length=len(prompt.split()) + MAX_ANSWER_TOKENS,
        min_length=1,
        # top_p=0.9,
        repetition_penalty=1.5,
        length_penalty=1.0,
        temperature=1,
    )
    generated_text = processor.batch_decode(generated_ids, skip_special_tokens=True)[0].strip()
    return generated_text

