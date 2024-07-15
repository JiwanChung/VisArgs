import torch

from transformers import AutoModelForCausalLM, LlamaTokenizer
from PIL import Image

DEVICE = 'cuda'
torch_type = torch.bfloat16

model = AutoModelForCausalLM.from_pretrained(
    "THUDM/cogvlm-chat-hf",
    torch_dtype=torch_type,
    low_cpu_mem_usage=True,
    load_in_4bit=False,
    trust_remote_code=True
).eval()
model.to(DEVICE)

tokenizer = LlamaTokenizer.from_pretrained("lmsys/vicuna-7b-v1.5")

def cogvlm(img, prompt):
    image = Image.open(img).convert("RGB")
    history = []
    
    if prompt is None:
        prompt = PROMPT
    input_by_model = model.build_conversation_input_ids(tokenizer, query=prompt, history=history, images=[image])
    
    inputs = {
        'input_ids': input_by_model['input_ids'].unsqueeze(0).to(DEVICE),
        'token_type_ids': input_by_model['token_type_ids'].unsqueeze(0).to(DEVICE),
        'attention_mask': input_by_model['attention_mask'].unsqueeze(0).to(DEVICE),
        'images': [[input_by_model['images'][0].to(DEVICE).to(torch_type)]] if image is not None else None,
    }
    if 'cross_images' in input_by_model and input_by_model['cross_images']:
        inputs['cross_images'] = [[input_by_model['cross_images'][0].to(DEVICE).to(torch_type)]]
        
    gen_kwargs = {
        "max_length": 2048,
        "temperature": 0.9,
        "do_sample": False
    }
    
    with torch.no_grad():
        outputs = model.generate(**inputs, **gen_kwargs)
        outputs = outputs[:, inputs['input_ids'].shape[1]:]
        response = tokenizer.decode(outputs[0])
        response = response.split("</s")[0]
    
    return response[0]
