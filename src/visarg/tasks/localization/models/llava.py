from PIL import Image
import transformers
import numpy as np

import torch

PRETRAINED_PATH = "llava-hf/llava-1.5-7b-hf"
DTYPE = torch.float16
DEVICE = 'cuda'

pipeline = transformers.pipeline(
    task='image-to-text',
    model=PRETRAINED_PATH,
    torch_dtype=DTYPE,
    device=DEVICE,
)

def llava(img, targets):
  # device = "cuda" if torch.cuda.is_available() else "cpu"
  # print(device)
  # model.to(device)
  if isinstance(img, str):
    img = Image.open(img)
  h, w, _ = np.array(img).shape
  bboxes = []
  for target in targets:
    prompt = f"<image>\nUSER:\nTarget:{target}\nGet a bounding box of the target in the image.\nASSISTANT:"
    generated_text = pipeline(
      images=img,
      prompt=prompt,
      generate_kwargs={"max_new_tokens": 200},
    )[0]['generated_text']
    raw_bbox = generated_text.split('ASSISTANT')[1].strip()
    bbox = []
    for i, d in enumerate(raw_bbox[raw_bbox.index('[')+1:raw_bbox.index(']')].split(',')):
      if i == 0 or i == 2:
        bbox.append(round(float(d.strip()) * w))
      else:
        bbox.append(round(float(d.strip()) * h))
    bboxes.append(bbox)
  return bboxes