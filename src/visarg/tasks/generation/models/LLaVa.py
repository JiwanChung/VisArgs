import transformers
import torch

PRETRAINED_PATH = "llava-hf/llava-1.5-7b-hf"
PROMPT = "<image>\nYour task is to answer what the image wants to say. You should answer in only one sentence without an unnecessary prefix. ANSWER:"
DEVICE = 'cuda'
MAX_ANSWER_TOKENS = 1024
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
        text = "<image>" + prefix + text.split("<image>")[-1]
        
    if postfix is not None:
        text = text + postfix
    
    return text

def llava(image_path, prompt=None):
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

    answer_word = prompt.split(' ')[-1]
    return generated_text.split(answer_word)[-1].strip()
