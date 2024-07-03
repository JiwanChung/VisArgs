from uio2.model import UnifiedIOModel
from uio2.runner import TaskRunner
from uio2.preprocessing import UnifiedIOPreprocessor
from uio2.preprocessing import build_batch

model = UnifiedIOModel.from_pretrained("allenai/uio2-xl")
preprocessor = UnifiedIOPreprocessor.from_pretrained("allenai/uio2-preprocessor", tokenizer="./llama_tokenizer.model")
runner = TaskRunner(model, preprocessor)

PROMPT = "\nYour task is to answer what the image wants to say. You should answer in only one sentence without an unnecessary prefix. ANSWER:"

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
# runner = TaskRunner(model, preprocessor)
# model.set_modalities(input_modalities=["text, image"], target_modalities=["text"])
def unifiedio2(img, prompt=PROMPT):
  # preprocessed_example = preprocessor(text_inputs="What color is the sky?", target_modality="text")
  # batch = build_batch([preprocessed_example], device=model.device)
  # tokens = model.generate(batch, modality="text", max_new_tokens=128)
  # print(tokens)
  # prep = preprocessor(image_inputs=img, text_inputs=prompt)
  # batch = build_batch([prep], device=model.device)
  # tokens = model.generate(batch, modality='text', max_new_tokens=128)
  
  answer = runner.vqa(img, prompt)
  return answer
