# Selective Vision is the Challenge for Visual Reasoning: A Benchmark for Visual Argument Understanding


<p align="center">
    <img src="./static/VisArg_Fig_1_PolarBear.png" alt=figure width=512px>
</p>

[Selective Vision is the Challenge for Visual Reasoning: A Benchmark for Visual Argument Understanding](https://arxiv.org/abs/2406.18925)

```
@article{chung2024selective,
  title={Selective Vision is the Challenge for Visual Reasoning: A Benchmark for Visual Argument Understanding},
  author={Chung, Jiwan and Lee, Sungjae and Kim, Minseo and Han, Seungju and Yousefpour, Ashkan and Hessel, Jack and Yu, Youngjae},
  journal={arXiv preprint arXiv:2406.18925},
  year={2024}
}
```

Please cite our work if you find our data helpful.

## Data

Our recommendation is to access the corpus on huggingface:

```python

from datasets import load_dataset

# load main data
dset = load_dataset("jiwan-chung/visargs", "annotations")

# load in the predefined negative sets for retrieval in the "Identification of Premises" task.
dset = load_dataset("jiwan-chung/visargs", "negatives")
```

Here's an example instance:

```
{
    'url': 'https://i.pinimg.com/originals/5e/7f/10/5e7f108728fb848eb8e3cccfdd62ef8f.jpg',
    'visual_premises': [
        'A small plant is growing inside a plastic bag.',
        'The bag contains a bit of soil.',
        'The bag is tied at the top, enclosing the plant.'
    ],
    'conclusion': 'The image represents the struggle of nature to survive in a human-made, constraining environment, highlighting the need for environmental awareness and protection.',
    'b_box': [
        {'h': 41, 'startX': 302, 'startY': 554, 'w': 72},
        {'h': 51, 'startX': 223, 'startY': 589, 'w': 229},
        {'h': 421, 'startX': 46, 'startY': 219, 'w': 407}
    ],
    'commonsense_premises': [
        'Plants require soil, water, light, and air to grow.',
        'Plastic bags are not a natural environment for plant growth and can restrict access to necessary resources.',
        'The act of enclosing the plant in a bag could symbolize suffocation or limitation of growth.'
    ],
    'reasoning_steps': [
        '(VP1, VP2, CP1 -> IC1): The small plant is growing, showing its resilience and need for natural resources.',
        "(VP3, CP2, CP3 -> IC2): The plastic bag enclosing the plant symbolizes human-imposed constraints on nature's growth and survival.",
        "(IC1, IC2 -> C): The image represents nature's struggle to survive in a constrained environment, emphasizing the importance of environmental protection."
    ]
}
```

## Usage

1. Installation

```
    pip install torch>=2.1
    pip install -r requirements.txt
    pip install -e .
```

2. Data preprocessing

```
    python src/visarg/others/preprocess.py
```

3. Evaluation

We provide three complementary tasks for assessing the machine capacity of visual argument understanding.


- Task 1 (*Localization of Premises*)
```
    python src/visarg/main.py --task 1 --model_name $MODELNAME --grounding_type "openset"
    python src/visarg/main.py --task 1 --model_name $MODELNAME --grounding_type "closedset"
```

- Task 2 (*Idenfication of Premises*)
```
    python src/visarg/main.py --task 2 --model_name $MODENAME
```

- Task 3 (*Deduction of Conclusion*)

```
    python src/visarg/main.py --task 3 --model_name $MODELNAME --condition 0 --prompt_style 0
```
