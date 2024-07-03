import clip

def load_model(model_name):
  model, preprocess = clip.load(model_name, device='cuda')
  return model, preprocess
