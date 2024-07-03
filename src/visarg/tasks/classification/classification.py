import os
import json

from tqdm import tqdm

from config import DATASET_REPO_ID, OUT_PATH
from visarg.others.prompts import identification_of_premises_prompt

from datasets import load_dataset

MODEL_CLASSES = {
  'llava': 'llava',
  'llavanext': 'llavanext',
  'instructblip': 'instructblip',
  'unifiedio2': 'unifiedio2',
  'qwenvl': 'qwenvl',
  'idefics2':'idefics2',
  'cogvlm':'cogvlm',
}

def load_model(model_name):
  model_name = model_name.lower()
  if model_name in MODEL_CLASSES:
      model_class = MODEL_CLASSES[model_name]
      module = __import__(f"visarg.tasks.classification.models.{model_class}", fromlist=[model_name])
      return getattr(module, model_name)
  raise ValueError(f"No model found for {model_name}")

def gen_option_prompt(options):
  option_prompt = ""
  for i, option in enumerate(options):
    if i == 0:
      option_prompt += "A. " + option + '\n'
    elif i == 1:
      option_prompt += "B. " + option + '\n'
    else:
      option_prompt += "C. " + option + '\n'

  return option_prompt

def parse_result(out):
  if out.lower() == 'a':
    return 0
  elif out.lower() == 'b':
    return 1
  elif out.lower() == 'c':
    return 2
  else:
    return -1

def identification_of_premises(model_name):
  model = load_model(model_name)
  random_global_scores = []
  clip_global_scores = []
  colbert_global_scores = []
  colbert_clip_global_scores = []
  semantic_global_scores = []

  random_local_scores = []
  clip_local_scores = []
  colbert_local_scores = []
  colbert_clip_local_scores = []
  semantic_local_scores = []
  
  data = load_dataset(DATASET_REPO_ID)['train']
  for item in tqdm(data, desc='Identification_of_premises'):
    
    if len(item['negative_sets']) < 0:
      continue
    
    image_path = item['image']
    random_scores = []
    clip_scores = []
    colbert_scores = []
    colbert_clip_scores = []
    semantic_scores = []

    questions = item['negative_sets']
    for question in questions:
      description = question['description']
      random_option_prompt = gen_option_prompt(question['easy_vp_options'])
      clip_option_prompt = gen_option_prompt(question['hard_clip_vp_options'])
      colbert_option_prompt = gen_option_prompt(question['hard_colbert_vp_options'])
      colbert_clip_option_prompt = gen_option_prompt(question['hard_colbert_clip_vp_options'])
      semantic_option_prompt = gen_option_prompt(question['hard_semantic_vp_options'])

      random_prompt = identification_of_premises_prompt(random_option_prompt, description)
      clip_prompt = identification_of_premises_prompt(clip_option_prompt, description)
      colbert_prompt = identification_of_premises_prompt(colbert_option_prompt, description)
      colbert_clip_prompt = identification_of_premises_prompt(colbert_clip_option_prompt, description)
      semantic_prompt = identification_of_premises_prompt(semantic_option_prompt, description)

      random_out = model(image_path, random_prompt)
      clip_out = model(image_path, clip_prompt)
      colbert_out = model(image_path, colbert_prompt)
      colbert_clip_out = model(image_path, colbert_clip_prompt)
      semantic_out = model(image_path, semantic_prompt)

      random_result = parse_result(random_out)
      clip_result = parse_result(clip_out)
      colbert_result = parse_result(colbert_out)
      colbert_clip_result = parse_result(colbert_clip_out)
      semantic_result = parse_result(semantic_out)

    
      random_score = 1 if random_result == question['easy_answer'] else 0
      clip_score = 1 if clip_result == question['hard_clip_answer'] else 0
      colbert_score = 1 if colbert_result == question['hard_colbert_answer'] else 0
      colbert_clip_score = 1 if colbert_clip_result == question['hard_colbert_clip_answer'] else 0
      semantic_score = 1 if semantic_result == question['hard_semantic_answer'] else 0

      random_global_scores.append(random_score)
      random_scores.append(random_score)
      clip_global_scores.append(clip_score)
      clip_scores.append(clip_score)
      colbert_global_scores.append(colbert_score)
      colbert_scores.append(colbert_score)
      colbert_clip_global_scores.append(colbert_clip_score)
      colbert_clip_scores.append(colbert_clip_score)
      semantic_global_scores.append(semantic_score)
      semantic_scores.append(semantic_score)

    random_local_scores.append(sum(random_scores)/len(random_scores))
    clip_local_scores.append(sum(clip_scores)/len(clip_scores))
    colbert_local_scores.append(sum(colbert_scores)/len(colbert_scores))
    colbert_clip_local_scores.append(sum(colbert_clip_scores)/len(colbert_clip_scores))
    semantic_local_scores.append(sum(semantic_scores)/len(semantic_scores))

  out_path = os.path.join(OUT_PATH, 'task2', f'{model_name}_result.json')
  if not os.path.exists(os.path.join(OUT_PATH, 'task2')):
    os.makedirs(os.path.join(OUT_PATH, 'task2'))
  with open(out_path, 'w') as f:
    total_result = {
      "random_global_scores" : sum(random_global_scores)/len(random_global_scores),
      "clip_global_scores" : sum(clip_global_scores)/len(clip_global_scores),
      "colbert_global_scores" : sum(colbert_global_scores)/len(colbert_global_scores),
      "colbert_clip_global_scores" : sum(colbert_clip_global_scores)/len(colbert_clip_global_scores),
      "semantic_global_scores" : sum(semantic_global_scores)/len(semantic_global_scores),
      "random_local_scores" : sum(random_local_scores)/len(random_local_scores),
      "clip_local_scores" : sum(clip_local_scores)/len(clip_local_scores),
      "colbert_local_scores" : sum(colbert_local_scores)/len(colbert_local_scores),
      "colbert_clip_local_scores" : sum(colbert_clip_local_scores)/len(colbert_clip_local_scores),
      "semantic_local_scores" : sum(semantic_local_scores)/len(semantic_local_scores),
    }
    json.dump(total_result, f)



             
      

