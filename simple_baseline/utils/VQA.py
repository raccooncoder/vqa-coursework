import jpeg4py
import cv2
import torch
from utils.preprocess import anno_open, questions_preproc
from torch.utils.data import Dataset
from albumentations.pytorch import ToTensor
from albumentations import (Compose, CenterCrop, VerticalFlip, RandomSizedCrop,
                            HorizontalFlip, HueSaturationValue, ShiftScaleRotate,
                            Resize, RandomCrop, Normalize, Rotate, Normalize)

h, w = 224, 224
transforms = {
            'train':Compose([
        Resize(h, w),
        Normalize(),
        ToTensor()
        ]),
            'val':Compose([
        Resize(h, w),
        Normalize(),
        ToTensor()
        ]),
            'test': Compose([
        Resize(h, w),
        Normalize(),
        ToTensor()
        ]),
}

class VQA(Dataset):
    def __init__(self, path, bag_of_words, ans_dict=None,
                transfrom=transforms, mode='train'): 
        self.path = path
        self.ans_dict = ans_dict
        self.questions = questions_preproc(path + '/questions/questions.json', bag_of_words, mode)
        self.annotations = anno_open(path + '/annotations/annotations.json')
        self.transform = transforms[mode]
        self.mode = mode

    def get_q_embed_size(self):
        return self.questions.shape[1] 
        
    def get_n_ans(self):
        return len(self.ans_dict)

    def __len__(self):
        return self.questions.shape[0]

    def __getitem__(self, idx):
        annotation = self.annotations['annotations'][idx]
        
        image_id = str(annotation['image_id'])
        try:
            image = jpeg4py.JPEG(self.path + 
                                '/images/COCO_' + self.mode + '2014_000000' + 
                                image_id.zfill(6) + '.jpg').decode()
        except:
            image = cv2.imread(self.path + 
                                '/images/COCO_' + self.mode + '2014_000000' + 
                                image_id.zfill(6) + '.jpg')
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        
        image = self.transform(image=image)['image']
        question = torch.FloatTensor(self.questions[idx])
        
        if self.mode != 'test':
            try: 
                answer = self.ans_dict[annotation['multiple_choice_answer']]
            except:
                answer = -1

            return (image, question, answer)
        else:
            return (image, question)