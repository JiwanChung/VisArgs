import json
import os

from config import DATASET_REPO_ID, IMAGE_PATH, OUT_PATH, ANNOTATION
from visarg.others.utils import interUnion, save_image
from tqdm import tqdm

from datasets import load_dataset

MODEL_CLASSES = {
  'lisa': 'lisa',
  'uninext_h': 'UNINEXT_H',
  'llava': 'llava',
  'ofa': 'ofa',
  'qwenvl': 'QwenVL',
  'lisa': 'lisa',
}

def load_model(model_name):
    model_name = model_name.lower()
    if model_name in MODEL_CLASSES:
        model_class = MODEL_CLASSES[model_name]
        module = __import__(f"visarg.tasks.localization.models.{model_class}", fromlist=[model_name])
        return getattr(module, model_name)
    raise ValueError(f"No model found for {model_name}")


def openset_grounding(model_name):
  data = load_dataset(DATASET_REPO_ID, data_files=ANNOTATION, split="train")
    
    print(f"== Openset Ground : {model_name} ==")
    
    ground = load_model(model_name)
    
    results = {}
    
    local_scores = []
    global_scores = []
    local_ious = []
    global_ious = []
    
    for idx, item in tqdm(enumerate(data), desc=f"Openset Grounding {model_name}"):
        local_score = []
        local_iou = []
        
        if 'b_box' not in item.keys():
            print("There is no bbox in data")
            continue
        
        image_path = save_image(item['url'])
        
        tgt_vps = []
        ground_bboxes = []
        
        # Filter out OCR problmes
        for i, vp in enumerate(item['visual_premises']):
            if '"' not in vp and 'text' not in vp and 'bubble' not in vp and 'logo' not in vp:
                tgt_vps.append(vp)
                ground_bboxes.append(item['b_box'][i])
        
        bboxes = ground(image_path, tgt_vps)
        
        image_result = {
            "vps": tgt_vps,
            "ious": [],
            "gts": [],
            "preds": [],
        }
        
        for pred, gt in zip(bboxes, ground_bboxes):
            gt_bbox = [
                gt["startX"],
                gt["startY"],
                gt["startX"] + gt["w"],
                gt["startY"] + gt["h"],
            ]
            
            iou = interUnion(pred, gt_bbox)
            
            image_result["ious"].append(iou)
            image_result["gts"].append(gt_bbox)
            image_result["preds"].append(pred)
            
            local_iou.append(iou)
            global_ious.append(iou)
            if iou > 0.5:
                local_score.append(1)
                global_scores.append(1)
            else:
                local_score.append(0)
                global_scores.append(0)
                
        results[idx] = image_result
        local_score = sum(local_score)/len(local_score)
        local_scores.append(local_score)
        local_iou = sum(local_iou)/len(local_iou)
        local_ious.append(local_iou)
                
        results[image_path] = image_result
        local_score = sum(local_score)/len(local_score)
        local_scores.append(local_score)
        local_iou = sum(local_iou)/len(local_iou)
        local_ious.append(local_iou)
        
    os.makedirs(os.path.join(OUT_PATH, "task1"), exist_ok=True)
    with open(os.path.join(OUT_PATH, "task1", f"{model_name}_result.json")) as r_file:
        json.dump(results, r_file)
        
    print('local iou : ', sum(local_ious)/len(local_ious))
    print('local score : ', sum(local_scores)/len(local_scores))
    print('global iou : ', sum(global_ious)/len(global_ious))
    print('global score : ', sum(global_scores)/len(global_scores))