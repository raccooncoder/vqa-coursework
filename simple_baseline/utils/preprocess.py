import numpy as np
import json
from tqdm import tqdm

def anno_open(path):
    with open(path, 'r') as file:
        annotations = json.load(file)
    
    return annotations

def create_ans_dict(path):
    with open(path, 'r') as file:
        annotations = json.load(file)

    print('Generating answer dictionary...')

    ans_list = []

    for a in tqdm(annotations['annotations']):
        ans_list.append(a['multiple_choice_answer'])

    ans_list = np.unique(ans_list)
    ans_dict = {v : k for k, v in enumerate(ans_list)}

    n_ans = len(ans_dict)

    print('Answer dictionary size: {} words.\n'.format(n_ans))
    return ans_dict

def create_bow(path, thresh):
    with open(path, 'r') as file:
        questions = json.load(file)

    print('Generating bag of words...')

    count = {}
    bag_of_words = {}
    questions = [q['question'].lower()[:-1].split(' ') for q in tqdm(questions['questions'])]

    for question in tqdm(questions):
        for word in question:
            if word not in count:
                count[word] = 1
            else:
                count[word] += 1

    for question in tqdm(questions):
        for word in question:
            if count[word] > thresh:
                if word not in bag_of_words:
                    bag_of_words[word] = len(bag_of_words) 

    n_words = len(bag_of_words)

    print('Bag of words size: {} words.\n'.format(n_words))

    return bag_of_words

def questions_preproc(path, bag_of_words, mode):
    with open(path, 'r') as file:
        questions = json.load(file)
    
    print('Preprocessing {} questions...'.format(mode))

    questions = [q['question'].lower()[:-1].split(' ') for q in tqdm(questions['questions'])]

    n_words = len(bag_of_words)
    result = np.zeros((len(questions), len(bag_of_words)))

    for i, question in enumerate(tqdm(questions)):
        q_vec = np.zeros(n_words)
        
        for word in question:
            if word in bag_of_words:
                q_vec[bag_of_words[word]] = 1
                
        result[i, :] = q_vec

    print('Done.\n')

    return result