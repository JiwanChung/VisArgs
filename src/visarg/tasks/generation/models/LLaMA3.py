import transformers
import torch

model_id = "meta-llama/Meta-Llama-3-8B-Instruct"
pipeline = transformers.pipeline(
    "text-generation",
    model=model_id,
    model_kwargs={"torch_dtype": torch.bfloat16},
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

def llama3(prompt=None):
    if prompt is None:
        prompt = PROMPT
    messages = [
        {"role": "system", "content": "You are a chatbot"},
        {"role": "user", "content": prompt}
    ]
    
    prompt_eng = pipeline.tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True,
    )
    
    terminators = [
        pipeline.tokenizer.eos_token_id,
        pipeline.tokenizer.convert_tokens_to_ids("<|eot_id|>")
    ]

    outputs = pipeline(
        prompt_eng,
        max_new_tokens=256,
        eos_token_id=terminators,
        do_sample=True,
        temperature=0.6,
        top_p=0.9,
    )
    return outputs[0]["generated_text"][len(prompt_eng):]
    
