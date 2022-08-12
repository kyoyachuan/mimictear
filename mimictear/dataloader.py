import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms

import json
import os
from PIL import Image

from .contants import IMAGES_ROOT, METADATAS_ROOT, LABEL_PATH, TRAIN_PATH


def random_generate_labels(size, cols, min=1, max=3):
    labels = torch.zeros(size, cols)
    for i in range(size):
        label_count = torch.randint(min, max, (1,))
        shuffle_idx = torch.randperm(cols)
        for s in shuffle_idx[:label_count]:
            labels[i, s] = 1
    return labels


def get_training_data(obj_dict):
    with open(TRAIN_PATH, 'r') as f:
        j = json.loads(f.read())
    img_name = []
    label = []
    for key, val in j.items():
        img_name.append(key)
        label.append([obj_dict[item] for item in val])
    return img_name, label


def get_test_label(file):
    with open(LABEL_PATH, 'r') as f:
        objects = json.loads(f.read())
    with open(f'{METADATAS_ROOT}/{file}', 'r') as f:
        test_list = json.loads(f.read())

    labels = torch.zeros(len(test_list), len(objects))
    for i in range(len(test_list)):
        for cond in test_list[i]:
            labels[i, int(objects[cond])] = 1
    return labels


class IclevrDataset(Dataset):
    def __init__(self, root=IMAGES_ROOT):

        self.root = root

        self.obj_dict = self.get_object_json()
        self.img_name, self.label = get_training_data(self.obj_dict)
        self.num_classes = len(self.obj_dict)

        self.transformations = transforms.Compose([
            transforms.Resize((64, 64)),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
        ])

    def __len__(self):
        return len(self.label)

    def __getitem__(self, index):
        img = Image.open(os.path.join(self.root, self.img_name[index])).convert('RGB')
        img = self.transformations(img)

        condition = self.label[index]
        one_hot_condition = torch.zeros(self.num_classes)
        for i in condition:
            one_hot_condition[i] = 1.

        return img, one_hot_condition

    def get_object_json(self):
        with open(LABEL_PATH, 'r') as f:
            j = json.loads(f.read())
        return j


def get_data(batch_size, num_workers):
    dataset = IclevrDataset()
    loader = DataLoader(dataset, shuffle=True, batch_size=batch_size, num_workers=num_workers)
    loader = iter(loader)

    while True:
        try:
            yield next(loader)

        except StopIteration:
            loader = DataLoader(
                dataset, shuffle=True, batch_size=batch_size, num_workers=num_workers
            )
            loader = iter(loader)
            yield next(loader)
