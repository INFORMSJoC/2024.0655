# %%
import os
import torch
from torch.utils.data import DataLoader, RandomSampler, SequentialSampler, Dataset
import numpy as np
import pickle
import json

import torchvision.transforms as transforms
import transformers
from PIL import Image
from PIL import ImageFile

ImageFile.LOAD_TRUNCATED_IMAGES = True

def load_pickle(path):
    with open(path, 'rb') as f:
        return pickle.load(f)

def load_json(path):
    with open(path, 'r') as f:
        return json.load(f)



def get_loader(args):
    if args.dataset == 'twitter':
        data_path = './datasets/twitter/'
        train_set = TwitterFeatureDataset(data_path, 'train')
        dev_set = TwitterFeatureDataset(data_path, 'test')
        test_set = TwitterFeatureDataset(data_path, 'test')
    elif args.dataset == 'weibo2':
        data_path = './datasets/weibo/'
        train_set = WeiboDataset(data_path, 'train')
        dev_set = WeiboDataset(data_path, 'validate')
        test_set = WeiboDataset(data_path, 'test')
    elif args.dataset == 'fakeddit':
        data_path = './datasets/public_images/'
        train_set = FakedditDataset(data_path, 'train')
        dev_set = FakedditDataset(data_path, 'validate')
        test_set = FakedditDataset(data_path, 'test')

    else:
        raise NotImplementedError

    train_sampler = RandomSampler(train_set)
    dev_sampler = SequentialSampler(dev_set)
    test_sampler = SequentialSampler(test_set)

    train_loader = DataLoader(train_set,
                              sampler=train_sampler,
                              batch_size=args.train_batch_size,
                              num_workers=args.num_workers,
                              pin_memory=True,)
    dev_loader = DataLoader(dev_set,
                            sampler=dev_sampler,
                            batch_size=args.train_batch_size,
                            num_workers=args.num_workers,
                            pin_memory=False) if dev_set is not None else None
    test_loader = DataLoader(test_set,
                             sampler=test_sampler,
                             batch_size=args.train_batch_size,
                             num_workers=args.num_workers,
                             pin_memory=False) if test_set is not None else None

    return train_loader, dev_loader, test_loader


class TwitterFeatureDataset(Dataset):
    def __init__(self, data_path, data_type):
        self.data_path = data_path
        self.data_type = data_type
        self.test_data_text, self.test_data_img, self.test_labels = self._get_data(data_type)
        print('>>', data_type, ' image feature: ', self.test_data_img.shape)
        print('>>', data_type, ' text feature: ', self.test_data_text.shape)

    def _get_data(self, data_type):
        data_text = np.load(self.data_path + data_type + "_text_with_label.npz")
        data_img = np.load(self.data_path + data_type + "_image_with_label.npz")
        text = torch.from_numpy(data_text["data"]).float()
        img = torch.from_numpy(data_img["data"]).squeeze().float()
        labels = torch.from_numpy(data_text["label"]).long()
        return text, img, labels

    def __len__(self):
        return self.test_data_text.shape[0]

    def __getitem__(self, item):
        return self.test_data_text[item], self.test_data_img[item], self.test_labels[item], item


vit_transform = transforms.Compose(
    [
        transforms.Resize(384),
        transforms.CenterCrop(384),
        transforms.ToTensor(),
        transforms.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225],
        ),
    ]
)

def vit_image_processor(image):
    return {"pixel_values": vit_transform(image)}


def WeiboDataset(data_path, data_type):
    with open(f"{data_path}{data_type}_datas.json", 'r') as f:
        datas = json.load(f)
    text_model = "hfl/chinese-roberta-wwm-ext"
    tokenizer = transformers.BertTokenizer.from_pretrained(text_model)
    return VIDataset(datas, data_path, tokenizer, vit_image_processor, max_len=512)


def FakedditDataset(data_path, data_type):
    datas = load_json(f"{data_path}{data_type}.json")
    text_model = "FacebookAI/roberta-base"
    tokenizer = transformers.RobertaTokenizer.from_pretrained(text_model)
    return VIDataset(datas, data_path, tokenizer, vit_image_processor, max_len=512)



class VIDataset(Dataset):
    def __init__(self, datas, data_path, tokenizer, image_processor, max_len=128):
        self.data_path = data_path
        self.tokenizer = tokenizer
        self.image_processor = image_processor
        self.datas = datas
        self.max_len = max_len

    def _get_text(self, item):
        text = self.datas[item]["text"]
        encoded_input = self.tokenizer(text, return_tensors='pt', max_length=self.max_len, padding="max_length", truncation=True)
        return encoded_input["input_ids"].squeeze(), encoded_input["attention_mask"].squeeze()

    def _get_label(self, item):
        label = self.datas[item]["label"]
        return label

    def _get_image(self, item):
        image_path = self.data_path + self.datas[item]["image"]
        image = Image.open(image_path)
        if image.mode not in ('RGB'):
            # image = image.convert('RGBA' if image.info.get("transparency", None) is not None
            #                       else 'RGB')
            image = image.convert('RGB')
        image = self.image_processor(image)
        return image["pixel_values"].squeeze()

    def __len__(self):
        return len(self.datas)

    def __getitem__(self, item):
        return *self._get_text(item), self._get_image(item), self._get_label(item), item
