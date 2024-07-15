import cv2
from transformers import AutoTokenizer, CLIPImageProcessor
import torch
import torch.nn.functional as F

import numpy as np

from visarg.tasks.localization.models.LISA.model.LISA import LISAForCausalLM
from visarg.tasks.localization.models.LISA.model.segment_anything.utils.transforms import ResizeLongestSide
from visarg.tasks.localization.models.LISA.model.llava.mm_utils import tokenizer_image_token

PRETRAINED_PATH = "xinlai/LISA-7B-v1"
TORCH_DTYPE = torch.bfloat16

tokenizer = AutoTokenizer.from_pretrained(PRETRAINED_PATH)
model = LISAForCausalLM.from_pretrained(
    PRETRAINED_PATH,
    low_cpu_mem_usage=True,
    vision_tower="openai/clip-vit-large-patch14",
    seg_token_idx=tokenizer("[SEG]", add_special_tokens=False).input_ids[0],
)
model.get_model().initialize_vision_modules(model.get_model().config)
vision_tower = model.get_model().get_vision_tower()
vision_tower.to(device='cuda', dtype=TORCH_DTYPE)
model = model.bfloat16().cuda()

model.eval()

clip_image_processor = CLIPImageProcessor.from_pretrained("openai/clip-vit-large-patch14")
transform = ResizeLongestSide(1024)

def preprocess(
    x,
    pixel_mean=torch.Tensor([123.675, 116.28, 103.53]).view(-1, 1, 1),
    pixel_std=torch.Tensor([58.395, 57.12, 57.375]).view(-1, 1, 1),
    img_size=1024,
) -> torch.Tensor:
    """Normalize pixel values and pad to a square input."""
    # Normalize colors
    x = (x - pixel_mean) / pixel_std
    # Pad
    h, w = x.shape[-2:]
    padh = img_size - h
    padw = img_size - w
    x = F.pad(x, (0, padw, 0, padh))
    return x

def lisa(image_path, targets):
    image_np = cv2.imread(image_path)
    image_np = cv2.cvtColor(image_np, cv2.COLOR_BGR2RGB)
    original_size_list = [image_np.shape[:2]]

    image_clip = (
        clip_image_processor.preprocess(image_np, return_tensors="pt")[
            "pixel_values"
        ][0]
        .unsqueeze(0)
        .cuda()
        .bfloat16()
    )

    image = transform.apply_image(image_np)
    resize_list = [image.shape[:2]]

    image = (
        preprocess(torch.from_numpy(image).permute(2, 0, 1).contiguous())
        .unsqueeze(0)
        .cuda()
        .bfloat16()
    )

    bboxes = []

    for target in targets:   
        input_ids = tokenizer_image_token("<image>\n" + target, tokenizer, return_tensors="pt")
        input_ids = input_ids.unsqueeze(0).cuda()
        
        output_ids, pred_masks = model.evaluate(
            image_clip,
            image,
            input_ids,
            resize_list,
            original_size_list,
            max_new_tokens=512,
            tokenizer=tokenizer,
        )
        
        output_ids = output_ids[0][output_ids[0] != -200]
    
        # text_output = tokenizer.decode(output_ids, skip_special_tokens=False)
        # text_output = text_output.replace("\n", "").replace("  ", " ")
        # print("text_output: ", text_output)

        bbox = None
        for i, pred_mask in enumerate(pred_masks):
            if pred_mask.shape[0] == 0:
                continue
            
            pred_mask = pred_mask.detach().cpu().numpy()[0]
            pred_mask = pred_mask > 0

            masked_img = image_np.copy()
            masked_img[pred_mask] = (
                image_np * 0.5
                + pred_mask[:, :, None].astype(np.uint8) * np.array([255, 0, 0]) * 0.5
            )[pred_mask]

            cv2.imwrite(f'{target[0]}.jpg', masked_img)
            points = np.argwhere(pred_mask)
            start_point, end_point = points[0], points[-1]
            bbox = list(start_point) + list(end_point)
        
        bboxes.append([0.0, 0.0, 0.0, 0.0] if bbox is None else bbox)
    return bboxes

