from PIL import Image
import transformers
from transformers import AutoProcessor, LlavaNextForConditionalGeneration
import torch

PRETRAINED_PATH = "llava-hf/llava-v1.6-mistral-7b-hf"  # llava-v1.6-vicuna-7b-hf
PROMPT = "[INST]<image>\nYour task is to answer what the image wants to say. You should answer in only one sentence without an unnecessary prefix. ANSWER:[/INST]" # A chat between a curious human and an artificial intelligence assistant. The assistant gives helpful, detailed, and polite answers to the human's questions. USER: <image>\nWhat is author's intention of this image? ASSISTANT:
DEVICE = 'cuda'
MAX_ANSWER_TOKENS = 128
DTYPE = torch.float16
FP16 = True

pipeline = transformers.pipeline(
    task='image-to-text',
    model=PRETRAINED_PATH,
    torch_dtype=DTYPE,
    device=DEVICE,
)

def prompt(prefix : str = None, postfix : str = None, need_base : bool = True):
    if need_base:
        text = PROMPT
    else:
        text = ""
    
    if prefix is not None:
        text = "[INST]<image>" + prefix + text.split("<image>")[-1]
        
    if postfix is not None:
        text = text.split("[/INST]")[0] + postfix + "[/INST]"
    
    return text

def llavanext(image_path, prompt=None):
    if prompt is None:
        prompt = PROMPT
    if "<image>" not in prompt:
        prompt = "<image>\n" + prompt
    
    if image_path is None:
        generated_text = pipeline(
            prompt=prompt,
            generate_kwargs={"max_length": 3072}
        )[0]['generated_text']
    else:  
        generated_text = pipeline(
            images=image_path,
            prompt=prompt,
            generate_kwargs={"max_length": 3072}
        )[0]['generated_text']
    
    return generated_text.split("[/INST]")[-1].strip()

