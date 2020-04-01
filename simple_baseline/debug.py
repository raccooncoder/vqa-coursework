import torch
import numpy as np
import json
from tqdm import tqdm
import jpeg4py 
from utils.preprocess import create_bow, create_ans_dict, questions_preproc

thresh = 128

ans_dict = create_ans_dict('VQAv2/train/annotations/annotations.json') 
bag_of_words = create_bow('VQAv2/train/questions/questions.json', thresh)

with open('VQAv2/val/annotations/annotations.json', 'r') as file:
    annotations = json.load(file)

idx = 104617

annotation = annotations['annotations'][idx]
questions = questions_preproc('VQAv2/val/questions/questions.json', bag_of_words, 'val')        

image_id = str(annotation['image_id'])
print('VQAv2/val' + 
                    '/images/COCO_' + 'val' + '2014_000000' + 
                    image_id.zfill(6) + '.jpg')
image = jpeg4py.JPEG('VQAv2/val' + 
                    '/images/COCO_' + 'val' + '2014_000000' + 
                    image_id.zfill(6) + '.jpg').decode()

question = torch.FloatTensor(questions[idx])    

try: 
    answer = ans_dict[annotation['multiple_choice_answer']]
except:
    answer = -1

print('VQAv2/val' + 
                    '/images/COCO_' + 'val' + '2014_000000' + 
                    image_id.zfill(6) + '.jpg')
print(image)
print(question)
print(answer)