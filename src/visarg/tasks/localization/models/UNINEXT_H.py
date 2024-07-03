import sys
import os

from detectron2.config import get_cfg
from detectron2.projects.uninext import add_uninext_config
from detectron2.data.detection_utils import read_image

from visarg.tasks.localization.models.uninext_predictor import UNINEXTPredictor
# sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../../../../..', 'third_party/UNINEXT/projects/UNINEXT')))

# from predictor import UNINEXTPredictor

def setup_cfg():
    cfg = get_cfg()
    add_uninext_config(cfg)
    
    cfg.merge_from_file('./visual_argument_experiments/configs/image_joint_vit_huge_32g.yaml')
    
    cfg.DATASETS.TEST = ("refcoco-unc-val", )
    cfg.freeze()
    return cfg

cfg = setup_cfg()
predictor = UNINEXTPredictor(cfg)

def uninext_h(img_path, targets):
    """
    img_path: path to image
    targets: list[str]
    """
    img = read_image(img_path, format="BGR")
    bboxes = []
    for target in targets:
        pred = predictor(img, 'grounding', target)
        bbox = list(pred['instances'].pred_boxes.tensor[0].cpu().numpy())
        bbox = [round(x) for x in bbox]
        bboxes.append(bbox)

    return bboxes
