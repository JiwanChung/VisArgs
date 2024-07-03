from transformers import AutoModelForCausalLM, AutoTokenizer
from transformers.generation import GenerationConfig
import torch
torch.manual_seed(1)

DEVICE='cuda'


tokenizer = AutoTokenizer.from_pretrained("Qwen/Qwen-VL", trust_remote_code=True)
model = AutoModelForCausalLM.from_pretrained("Qwen/Qwen-VL", trust_remote_code=True).to(DEVICE)

def qwenvl(img, prompt):
  query = tokenizer.from_list_format([
    {'image': img},
    {'text': prompt},
  ])
  inputs = tokenizer(query, return_tensors='pt')
  inputs = inputs.to(model.device)
  pred = model.generate(**inputs)
  response = tokenizer.decode(pred.cpu()[0], skip_special_tokens=False)
  return response.split('ANSWER:')[1][0]
