from uio2.model import UnifiedIOModel
from uio2.runner import TaskRunner
from uio2.preprocessing import UnifiedIOPreprocessor
from uio2.preprocessing import build_batch


model = UnifiedIOModel.from_pretrained("allenai/uio2-xl")
preprocessor = UnifiedIOPreprocessor.from_pretrained("allenai/uio2-preprocessor", tokenizer="llama_tokenizer.model")

def unifiedio2(img, prompt):
  runner = TaskRunner(model, preprocessor)
  answer = runner.vqa(img, prompt)
  return answer[0]

