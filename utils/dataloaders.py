import os
import torch.utils.data as data
from PIL import Image
from utils import transforms as tr


'''
Load all training and validation data paths
'''
def full_path_loader(data_dir):
    train_data = [i for i in os.listdir(data_dir + 'train/A/') if not
    i.startswith('.')]
    train_data.sort()

    valid_data = [i for i in os.listdir(data_dir + 'val/A/') if not
    i.startswith('.')]
    valid_data.sort()

    train_label_paths = []
    val_label_paths = []
    for img in train_data:
        train_label_paths.append(data_dir + 'train/label/' + img)
    for img in valid_data:
        val_label_paths.append(data_dir + 'val/label/' + img)


    train_data_path = []
    val_data_path = []

    for img in train_data:
        train_data_path.append([data_dir + 'train/', img])
    for img in valid_data:
        val_data_path.append([data_dir + 'val/', img])

    train_dataset = {}
    val_dataset = {}
    for cp in range(len(train_data)):
        train_dataset[cp] = {'image': train_data_path[cp],
                         'label': train_label_paths[cp]}
    for cp in range(len(valid_data)):
        val_dataset[cp] = {'image': val_data_path[cp],
                         'label': val_label_paths[cp]}


    return train_dataset, val_dataset

'''
Load all testing data paths
'''
def full_test_loader(data_dir):

    test_data = [i for i in os.listdir(data_dir + 'test/A/') if not
                    i.startswith('.')]
    test_data.sort()

    test_label_paths = []
    for img in test_data:
        test_label_paths.append(data_dir + 'test/label/' + img)

    test_data_path = []
    for img in test_data:
        test_data_path.append([data_dir + 'test/', img])

    test_dataset = {}
    for cp in range(len(test_data)):
        test_dataset[cp] = {'image': test_data_path[cp],
                           'label': test_label_paths[cp]}

    return test_dataset

def cdd_loader(img_path, label_path, aug):
    dir = img_path[0]
    name = img_path[1]

    img1 = Image.open(dir + 'A/' + name)
    img2 = Image.open(dir + 'B/' + name)
    label = Image.open(label_path)
    sample = {'image': (img1, img2), 'label': label}

    if aug:
        sample = tr.train_transforms(sample)
    else:
        sample = tr.test_transforms(sample)

    return sample['image'][0], sample['image'][1], sample['label']


class CDDloader(data.Dataset):

    def __init__(self, full_load, aug=False):

        self.full_load = full_load
        self.loader = cdd_loader
        self.aug = aug

    def __getitem__(self, index):

        img_path, label_path = self.full_load[index]['image'], self.full_load[index]['label']

        return self.loader(img_path,
                           label_path,
                           self.aug)

    def __len__(self):
        return len(self.full_load)


# MOBILE_CDN_BCDD_PREPROCESSING
import cv2
import numpy as np
from utils import mobile_cdnet_transforms as mobile_tr

_mobile_mean = [0.406, 0.456, 0.485, 0.406, 0.456, 0.485]
_mobile_std = [0.225, 0.224, 0.229, 0.225, 0.224, 0.229]
_mobile_train_transform = mobile_tr.Compose([
    mobile_tr.Normalize(mean=_mobile_mean, std=_mobile_std),
    mobile_tr.Scale(256, 256),
    mobile_tr.RandomCropResize(int(7.0 / 224.0 * 256)),
    mobile_tr.RandomFlip(),
    mobile_tr.RandomExchange(),
    mobile_tr.ToTensor(),
])
_mobile_eval_transform = mobile_tr.Compose([
    mobile_tr.Normalize(mean=_mobile_mean, std=_mobile_std),
    mobile_tr.Scale(256, 256),
    mobile_tr.ToTensor(),
])

# This redefinition is deliberately below the original loader: CDDloader resolves
# cdd_loader at call time, so every BCDD sample now follows Mobile-CDNet exactly.
def cdd_loader(img_path, label_path, aug):
    directory, name = img_path
    image_t1 = cv2.imread(directory + 'A/' + name, cv2.IMREAD_COLOR)
    image_t2 = cv2.imread(directory + 'B/' + name, cv2.IMREAD_COLOR)
    label = cv2.imread(label_path, cv2.IMREAD_GRAYSCALE)
    assert image_t1 is not None and image_t2 is not None and label is not None, name
    image = np.concatenate((image_t1, image_t2), axis=2)
    image, label = (_mobile_train_transform if aug else _mobile_eval_transform)(image, label)
    return image[:3], image[3:], label
