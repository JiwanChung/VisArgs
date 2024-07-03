from modelscope.pipelines import pipeline
from modelscope.utils.constant import Tasks
from modelscope.preprocessors.multi_modal import OfaPreprocessor
model = 'damo/ofa_visual-question-answering_pretrain_large_en'
preprocessor = OfaPreprocessor(model_dir=model)
ofa_pipe = pipeline(
    Tasks.visual_question_answering,
    model=model,
    model_revision='v1.0.1',
    preprocessor=preprocessor)

PROMPT = "Your task is to answer what the image wants to say. You should answer in only one sentence without an unnecessary prefix. ANSWER:"

def prompt(prefix : str = None, postfix : str = None, need_base : bool = True):
    if need_base:
        text = PROMPT
    else:
        text = ""
    
    if prefix is not None:
        text = prefix + text
        
    if postfix is not None:
        text = text + postfix
    
    return text
  
def ofa(img, prompt=None):
  if prompt is None:
    prompt = PROMPT

  input = {'image': img, 'text': prompt}

  answer = ofa_pipe(input)
  answer = answer['text'][0]

  return answer
