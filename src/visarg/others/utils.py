import requests
from io import BytesIO
from PIL import Image

def interUnion(boxA, boxB):
  xA = max(boxA[0], boxB[0])
  yA = max(boxA[1], boxB[1])
  xB = min(boxA[2], boxB[2])
  yB = min(boxA[3], boxB[3])

  interArea = max(0, xB-xA+1) * max(0, yB-yA+1)
  AArea = (boxA[2]-boxA[0]+1) * (boxA[3]-boxA[1]+1)
  BArea = (boxB[2]-boxB[0]+1) * (boxB[3]-boxB[1]+1)

  iou = interArea/(AArea + BArea - interArea)

  return iou

def save_image(url):
  res = requests.get(url)
  img = Image.open(BytesIO(res.content))
  img_ext = url.split('.')[-1]
  if '?' in img_ext:
    img_ext = img_ext.split('?')[0]
  img_path = './temp.' + img_ext
  img.save(img_path)
  return img_path