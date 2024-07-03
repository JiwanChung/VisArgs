from modelscope.pipelines import pipeline
from modelscope.utils.constant import Tasks
from modelscope.outputs import OutputKeys
import warnings

warnings.filterwarnings('ignore')

def ofa(img, targets):
  ofa_pipe = pipeline(
      Tasks.visual_grounding,
      model='damo/ofa_visual-grounding_refcoco_large_en')
  
  bboxes = []
  for target in targets:
    prompt = target
    result = ofa_pipe(
      {
        'image': img,
        'text': prompt
      })
    bboxes.append(result[OutputKeys.BOXES][0])

  return bboxes