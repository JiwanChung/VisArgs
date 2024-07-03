import transformers
import torch

pipeline = transformers.pipeline(
    "text-generation",
    model="mistralai/Mistral-7B-Instruct-v0.2",
    model_kwargs={"torch_dtype": torch.float16},
    device="cuda",
)

PROMPT = "Your task is answer what is the image want to say. You should response only one sentence without unnecessary prefix. ANSWER:"

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

def mistral(prompt=None):
    if prompt is None:
        prompt = PROMPT

    outputs = pipeline(
        prompt,
        max_new_tokens=256,
    )
    return outputs[0]["generated_text"][len(prompt):].split("---")[0].strip("\n")

