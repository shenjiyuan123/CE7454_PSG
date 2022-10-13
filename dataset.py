import io
import json
import logging
import os

import torch
import torchvision.transforms as trn
from PIL import Image, ImageFile
from torch.utils.data import Dataset

# to fix "OSError: image file is truncated"
ImageFile.LOAD_TRUNCATED_IMAGES = True


class Convert:
    def __init__(self, mode='RGB'):
        self.mode = mode

    def __call__(self, image):
        return image.convert(self.mode)


def get_transforms(stage: str):
    mean, std = [0.48145466, 0.4578275, 0.40821073], [0.26862954, 0.26130258, 0.27577711]
    if stage == 'train':
        return trn.Compose([
            Convert('RGB'),
            trn.Resize((336, 336)),
            trn.RandomHorizontalFlip(),
            # trn.RandAugment(2,9),
            # trn.AutoAugment(),
            trn.ToTensor(),
            trn.Normalize(mean, std),
        ])

    elif stage in ['val', 'test']:
        return trn.Compose([
            Convert('RGB'),
            trn.Resize((336, 336)),
            trn.ToTensor(),
            trn.Normalize(mean, std),
        ])


class PSGClsDataset(Dataset):
    def __init__(
        self,
        stage,
        preprocess=None,
        root='./data/coco/',
        num_classes=56,
    ):
        super(PSGClsDataset, self).__init__()
        with open('./data/psg/psg_cls_basic.json') as f:
            dataset = json.load(f)
        self.imglist = [
            d for d in dataset['data']
            if d['image_id'] in dataset[f'{stage}_image_ids']
        ]
        self.root = root
        self.transform_image = get_transforms(stage)
        self.preprocess = preprocess
        self.num_classes = num_classes

    def get_all_label_for_sampler(self):
        label_all = torch.Tensor(self.__len__(), self.num_classes)
        label_all.fill_(0)
        for i, sample in enumerate(self.imglist):
            label_all[i][sample['relations']] = 1
        return label_all
        
    
    def __len__(self):
        return len(self.imglist)

    def __getitem__(self, index):
        sample = self.imglist[index]
        path = os.path.join(self.root, sample['file_name'])
        try:
            with open(path, 'rb') as f:
                content = f.read()
                filebytes = content
                buff = io.BytesIO(filebytes)
                image = Image.open(buff).convert('RGB')
                sample['data'] = self.transform_image(image)
                # make the image into patch
                # sample['data'] = self.preprocess(image)
        except Exception as e:
            logging.error('Error, cannot read [{}]'.format(path))
            raise e
        # Generate Soft Label
        soft_label = torch.Tensor(self.num_classes)
        soft_label.fill_(0)
        soft_label[sample['relations']] = 1
        sample['soft_label'] = soft_label
        # del sample['relations']
        out = {
            'image_id':sample['image_id'],
            'file_name':sample['file_name'],
            'data':sample['data'],
            'soft_label':sample['soft_label']
            }
        return out


