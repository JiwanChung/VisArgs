import json

from visarg.tasks.localization.closedset import closedset_grounding
from visarg.tasks.localization.openset import openset_grounding

def localization_of_premises(grounding_type, model_name):
  if grounding_type == 'closedset':
    closedset_grounding(model_name)
  else:
    openset_grounding(model_name)