import torch
from PIL import Image
from io import BytesIO

from transformers import AutoProcessor, AutoModelForVision2Seq

PROMPT = "<image>\nYour task is to answer what the image wants to say. You should answer in only one sentence without an unnecessary prefix. ANSWER:"
DEVICE = "cuda"

model = AutoModelForVision2Seq.from_pretrained(
    "HuggingFaceM4/idefics2-8b-chatty",
    torch_dtype=torch.float16,
    _attn_implementation="flash_attention_2",
).to(DEVICE)

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

processor = AutoProcessor.from_pretrained(
    "HuggingFaceM4/idefics2-8b",
)

def idefics2(image_path, prompt=None):
    if prompt is None:
        prompt = PROMPT
    
    if "<image>" not in prompt:
        prompt = "<image>" + prompt
        
    if isinstance(image_path, list):
        image = [image_path]
        
        num_ex_image = len(image_path)-1
        for i in range(num_ex_image):
            prompt = prompt.replace(f'(Example {str(i+1)})', f'<image>\n(Example {str(i+1)})')
    else:
        image = Image.open(image_path)
        
    # print(prompt)
    inputs = processor(images=image, text=prompt, return_tensors="pt")
    
    inputs = {k: v.to(DEVICE) for k, v in inputs.items()}

    # Generate
    generated_ids = model.generate(**inputs, max_new_tokens=500)
    generated_texts = processor.batch_decode(generated_ids, skip_special_tokens=True)[0]

    answer_word = prompt.split(' ')[-1]
    return generated_texts.split(answer_word)[-1].replace('GroupLayout: ', '').strip('\n')

