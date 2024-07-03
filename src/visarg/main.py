import argparse
from distutils.util import strtobool

from visarg.tasks.localization.localization import localization_of_premises
from visarg.tasks.classification import identification_of_premises
from visarg.tasks.generation.generation import deduction_of_conclusion

def parse_args():
  parser = argparse.ArgumentParser()

  # Common
  parser.add_argument('--task', type=int, default=1)
  parser.add_argument('--model_name', type=str, default='llava')
  
  # Grounding (task1)
  parser.add_argument('--grounding_type', type=str, default='openset')
  
  # Conclusion deduction options (task3)
  parser.add_argument('--condition', type=int, default=0, choices=range(4))
  parser.add_argument('--prompt_style', type=int, default=0, choices=range(4))

  parser.add_argument('--text2con', action="store_true")

  args = parser.parse_args()
  return args

if __name__ == "__main__":
  args = parse_args()
  task = args.task

  if task == 1:
    localization_of_premises(args.grounding_type, args.model_name)
  if task == 2:
    identification_of_premises(args.model_name)
  if task == 3:
    deduction_of_conclusion(args)
