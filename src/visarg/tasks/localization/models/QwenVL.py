from transformers import AutoModelForCausalLM, AutoTokenizer
import re

model = AutoModelForCausalLM.from_pretrained("Qwen/Qwen-VL", device_map="cuda", trust_remote_code=True).eval()

tokenizer = AutoTokenizer.from_pretrained("Qwen/Qwen-VL", trust_remote_code=True)

PATTERN = re.compile(r'\((.*?)\),\((.*?)\)')

def qwenvl(img, targets):
    responses = []
    for idx, target in enumerate(targets):
        # print(idx, target)
        query = tokenizer.from_list_format([
            {'image': img},
            {'text': f"<img>{img}</img><ref>{target}</ref><box>"}
        ])

        inputs = tokenizer(query, return_tensors='pt')
        inputs = inputs.to(model.device)
        pred = model.generate(**inputs)
        response = tokenizer.decode(pred.cpu()[0], skip_special_tokens=False)
        image = tokenizer.draw_bbox_on_latest_picture(response)
        # if image is not None:
        #     image.save(f'{idx}.jpg')
        predict_bbox = re.findall(PATTERN, response)
        try:
            if ',' not in predict_bbox[0][0] or ',' not in predict_bbox[0][
                    1]:
                predict_bbox = (0., 0., 0., 0.)
            else:
                x1, y1 = [
                    float(tmp) for tmp in predict_bbox[0][0].split(',')
                ]
                x2, y2 = [
                    float(tmp) for tmp in predict_bbox[0][1].split(',')
                ]
                predict_bbox = (x1, y1, x2, y2)
        except:
            predict_bbox = (0., 0., 0., 0.)
        
        responses.append(predict_bbox)

    return responses


    