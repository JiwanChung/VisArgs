import os
import torch
import json
import copy
import clip
from tqdm import tqdm
from PIL import Image, ImageDraw
from datasets import load_dataset

from config import DATASET_PATH, IMAGE_PATH, OUT_PATH, DATASET_REPO_ID, ANNOTATION
from visarg.others.utils import interUnion, save_image
from visarg.tasks.localization.models.clip import load_model

def closedset_grounding(model_name):
  model, preprocess = load_model(model_name)

  data = load_dataset(DATASET_REPO_ID, data_files=ANNOTATION, split="train")
  scores = []
  for item in tqdm(data, desc='closedset grounding'):
    image_path = save_image(item['url'])
    img = Image.open(image_path)
    vps = item['visual_premises']
    local_boxes = item['b_box']

    filtered_texts = []
    filtered_boxes = []

    for vp_i, vp in enumerate(vps):
      if '"' not in vp and 'text' not in vp and 'bubble' not in vp and 'logo' not in vp:
        filtered_texts.append(vp)
        filtered_boxes.append(local_boxes[vp_i])

    local_scores = []
    if len(filtered_texts) > 0:  
      vp_features = []
      box_features = []
      with torch.no_grad():
        for vp, box in zip(filtered_texts, filtered_boxes):
          tgt_img = img.crop((box['startX'], box['startY'], box['startX'] + box['w'], box['startY'] + box['h']))
          image_input = preprocess(tgt_img).unsqueeze(0).to('cuda')
          text_input= clip.tokenize(vp[:300]).to('cuda')

          image_features = model.encode_image(image_input)
          text_features = model.encode_text(text_input)
          image_features /= image_features.norm(dim=-1, keepdim=True)
          text_features /= text_features.norm(dim=-1, keepdim=True)

          vp_features.append(text_features)
          box_features.append(image_features)

      vp_features = torch.cat(vp_features, dim=0)
      box_features = torch.cat(box_features, dim=0)

      for vp_i, vp in enumerate(filtered_texts):
        similarity = (vp_features[vp_i] @ box_features.T).softmax(dim=-1)
        pred_box = torch.argmax(similarity)
        
        pred_box = filtered_boxes[pred_box]
        gt_box = filtered_boxes[vp_i]

        target_box_T = [pred_box['startX'], pred_box['startY'], pred_box['startX'] + pred_box['w'], pred_box['startY'] + pred_box['h']]
        gt_box_T = [gt_box['startX'], gt_box['startY'], gt_box['startX'] + gt_box['w'], gt_box['startY'] + gt_box['h']]

        if interUnion(target_box_T, gt_box_T) > 0.5:
          local_scores.append(1)
        else:
          local_scores.append(0)

      scores.append(sum(local_scores)/len(local_scores)) 

  model_name = model_name.replace('/', '_')
  out_path = os.path.join(OUT_PATH, 'task1', f'{model_name}_closedset_result.json')
  if not os.path.exists(os.path.join(OUT_PATH, 'task1')):
    os.makedirs(os.path.join(OUT_PATH, 'task1'))  
  with open(out_path, 'w') as f:
    result = {
      'scores': sum(scores)/len(scores)
    }
    json.dump(result, f)