from transformers import AutoModelForCausalLM, AutoTokenizer

model = AutoModelForCausalLM.from_pretrained("Qwen/Qwen-VL-Chat", device_map="cuda", trust_remote_code=True).eval()

tokenizer = AutoTokenizer.from_pretrained("Qwen/Qwen-VL-Chat", trust_remote_code=True)

PROMPT = "Question: Your task is to answer what the image wants to say. You should answer in only one sentence without an unnecessary prefix. ANSWER:"

def prompt(prefix : str = None, postfix : str = None, need_base : bool = True):
    if need_base:
        text = PROMPT
    else:
        text = ""
    
    if prefix is not None:
        text = "Question: " + prefix + text.split("Question: ")[-1]
        
    if postfix is not None:
        text = text + postfix
    
    return text

def qwenvlchat(img, prompt=None):
    if prompt is None:
        prompt = PROMPT
    query = tokenizer.from_list_format([
        {'image': img},
        {'text': prompt}
    ])

    response, history = model.chat(tokenizer, query=query, history=None)
    
    return response
