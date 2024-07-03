from config import (DATASET_REPO_ID, IMAGE_PATH, OUT_PATH)

import json
import os
import nltk

nltk.download('punkt')

from tqdm import tqdm

from visarg.others.prompt_styles import PROMPT_STYLES

from datasets import load_dataset

MODEL_CLASSES = {
  'blip2': 'BLIP2',
  'instructblip': 'InstructBLIP',
  'kosmos2': 'KOSMOS2',
  'llavanext': 'LLaVANeXT',
  'llava': 'LLaVa',
  'cogvlm': 'CogVLM',
  'qwenvlchat': 'QwenVLChat',
  'minigpt_4': 'MiniGPT_4',
  'openflamingo': 'Openflamingo',
  'ofa': 'ofa',
  'idefics2': 'idefics2',
  'otter': 'Otter',
  'unifiedio2': 'unifiedio2',
  'llama3': 'LLaMA3',
  'llama2': 'LLaMA2',
  'mistral': 'Mistral',
  'zephyr': 'Zephyr',
}


def load_model(model_name):
  model_name = model_name.lower()
  if model_name in MODEL_CLASSES:
      model_class = MODEL_CLASSES[model_name]
      module = __import__(f"visarg.tasks.generation.models.{model_class}", fromlist=[model_name])
      return getattr(module, model_name)
  raise ValueError(f"No model found for {model_name}")


def load_prompt_func(model_name):
  model_name = model_name.lower()
  if model_name in MODEL_CLASSES:
    model_class = MODEL_CLASSES[model_name]
    module = __import__(f"visarg.tasks.generation.models.{model_class}", fromlist=["prompt"])
    return getattr(module, "prompt")
  raise ValueError(f"No model's prompt function found for {model_name}")


def get_prompt(prompt_func, condition, prompt_style, vps, cps, rs):
  need_base = False
  if condition:
    if condition == 1:
      # Image and vps -> Conclusion
      description = PROMPT_STYLES[prompt_style]["vp_desc"]
          
      informations = '\n\n(Task Part)\n' + '\n'.join(vps) + '\n\n'
      prefix = description + informations
      
    elif condition == 2:
      # Image and cps -> Conclusion
      description = PROMPT_STYLES[prompt_style]["cp_desc"]
      informations = '\n\n(Task Part)\n' + '\n'.join(cps) + '\n\n'
      prefix = description + informations
    
    elif condition == 3:
      # Image and vps, cps -> Conclusion
      description = PROMPT_STYLES[prompt_style]["vp_desc"] + PROMPT_STYLES[prompt_style]["cp_desc"]
      informations = '\n\n(Task Part)\n' + '\n'.join(vps) + '\n\n' + '\n'.join(cps) + '\n\n'
      prefix = description + informations
      
    elif condition == 4:
      # Image and vps, cps, reasoning steps -> Conclusion
      description = PROMPT_STYLES[prompt_style]["vp_desc"] + PROMPT_STYLES[prompt_style]["cp_desc"] + PROMPT_STYLES[prompt_style]["rs_desc"]
      informations = '\n\n(Task Part)\n' + '\n'.join(vps) + '\n\n' + '\n'.join(cps) + '\n\n'
      rs_lines = '\n'.join(rs)
      rs_lines = rs_lines.split("-> C):")[0] + "-> C)" + "\n\n"
      prefix = description + informations + rs_lines
    
    if need_base:
      postfix = None
    else:
      postfix = PROMPT_STYLES[prompt_style]["task_desc"]

    prompt = prompt_func(prefix=prefix, postfix=postfix, need_base=need_base)
    return prompt


def deduction_of_conclusion(args):
  
  data = load_dataset(DATASET_REPO_ID)['train']

  print(f'== Load Model : {args.model_name} ==')
  print(f'= Prompt Style {args.prompt_style}')
  print(f'= Condition {args.condition}')

  deduct = load_model(args.model_name)
  prompt_func = load_prompt_func(args.model_name)

  gts = {}
  res = {}
  for idx, item in tqdm(enumerate(data), desc='Deduct conclusion', leave=True):
    vps = ["Visual Premises (VP):"] + [str(idx+1) + ". " + vp for idx, vp in enumerate(item['visual_premises'])]
    cps = ["Commonsense Premises (CP):"] + [str(idx+1) + ". " + cp for idx, cp in enumerate(item['commonsense_premises'])]
    rs = ["Reasoning Step:"] + item['reasoning_steps']

    image_path = data['image']

    prompt = get_prompt(prompt_func, args.condition, args.prompt_style, vps, cps, rs)

    try:
      if not args.text2con:
        con = deduct(image_path, prompt)
      else:
        con = deduct(prompt)
    except Exception as e:
      print(f"Exception {e} occured")

    try:
      con = nltk.tokenize.sent_tokenize(con)[0]
    except:
      con = con
    
    gts[idx] = item['reasoning_steps'][-1].split('-> C): ')[-1]
    res[idx] = con

  out_path = os.path.join(args.OUT_PATH, 'task3')
  os.makedirs(out_path, exist_ok=True)

  file_name = args.model_name.lower() + '_' + str(args.condition) + "_" + str(args.prompt_style)

  gts_path = os.path.join(out_path, file_name + '_gts.json')
  res_path = os.path.join(out_path, file_name + '_res.json')

  with open(gts_path, 'w') as f:
    json.dump(gts, f)
  with open(res_path, 'w') as f:
    json.dump(res, f)