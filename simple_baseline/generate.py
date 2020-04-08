'''
Generate question text corpus.
'''

import json
from tqdm import tqdm

path = 'VQAv2/train/questions/questions.json'

with open(path, 'r') as file:
    questions = json.load(file)

questions = [q['question'] for q in tqdm(questions['questions'])]

with open('questions.txt', 'w') as f:
    for q in tqdm(questions):
        f.write("{}\n".format(q))

