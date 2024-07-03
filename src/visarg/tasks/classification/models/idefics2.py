import torch
from PIL import Image
from io import BytesIO

from transformers import AutoProcessor, AutoModelForVision2Seq

DEVICE = "cuda"

model = AutoModelForVision2Seq.from_pretrained(
    "HuggingFaceM4/idefics2-8b-chatty",
    torch_dtype=torch.float16,
    _attn_implementation="flash_attention_2",
).to(DEVICE)

processor = AutoProcessor.from_pretrained(
    "HuggingFaceM4/idefics2-8b-chatty",
)    


def idefics2(image_path, prompt):
    
    if "<image>" not in prompt:
        prompt = "<image>" + prompt
    
    if isinstance(image_path, str):
        image = Image.open(image_path)
    else:
        image = image_path
    inputs = processor(images=image, text=prompt, return_tensors="pt")
    
    inputs = {k: v.to(DEVICE) for k, v in inputs.items()}

    generated_ids = model.generate(**inputs, max_new_tokens=500)
    generated_texts = processor.batch_decode(generated_ids, skip_special_tokens=True)[0]
    return generated_texts.split('ANSWER:')[1].split('\n')[0][0]
