"""load cifar10 dataset pytorch"""
import os
import os.path
from pathlib import Path
from typing import List
import numpy as np
import torch
import torch.utils.data as data
import torchvision.transforms as transforms
from PIL import Image
from torch.utils.data import random_split
from torchvision.datasets import CIFAR10
from torchvision.datasets.utils import download_url, check_integrity
from torch.utils.data import DataLoader

_CIFAR_TEST_TRANSFORM = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
])


def load_new_test_data(root: str, filename: str = 'cifar10.1'):
    """ Return [np.array, np.array] """
    data_path = root
    label_filename = filename + '-labels.npy'
    imagedata_filename = filename + '-data.npy'
    label_filepath = os.path.join(data_path, label_filename)
    imagedata_filepath = os.path.join(data_path, imagedata_filename)
    labels = np.load(label_filepath).astype(np.int64)
    imagedata = np.load(imagedata_filepath)
    return imagedata, labels


class CIFAR10_1(data.Dataset):
    images_url = 'https://github.com/modestyachts/CIFAR-10.1/blob/master/datasets/cifar10.1_v4_data.npy?raw=true'
    images_md5 = '29615bb88ff99bca6b147cee2520f010'
    images_md5 = None
    images_filename = 'cifar10.1-data.npy'

    labels_url = 'https://github.com/modestyachts/CIFAR-10.1/blob/master/datasets/cifar10.1_v4_labels.npy?raw=true'
    labels_md5 = 'a27460fa134ae91e4a5cb7e6be8d269e'
    labels_md5 = None
    labels_filename = 'cifar10.1-labels.npy'

    classes = [
        'airplane', 'automobile', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse',
        'ship', 'truck'
    ]

    @property
    def targets(self):
        return self.labels

    def __init__(self,
                 root,
                 transform=None,
                 target_transform=None,
                 download=False):
        self.root = os.path.expanduser(root)
        self.transform = transform
        self.target_transform = target_transform

        if download:
            self.download()

        if not self._check_integrity():
            raise RuntimeError('Dataset not found or corrupted.' +
                               ' You can use download=True to download it')

        images, labels = load_new_test_data(root)

        self.data = images
        self.labels = labels

        self.class_to_idx = {
            _class: i
            for i, _class in enumerate(self.classes)
        }

    def __getitem__(self, index):
        """
        Args:
            index (int): Index
        Returns:
            tuple: (image, target) where target is index of the target class.
        """
        img, target = self.data[index], self.labels[index]

        # doing this so that it is consistent with all other datasets
        # to return a PIL Image
        img = Image.fromarray(img)

        if self.transform is not None:
            img = self.transform(img)

        if self.target_transform is not None:
            target = self.target_transform(target)

        return img, target

    def __len__(self):
        return len(self.data)

    def _check_integrity(self):
        data_path = os.path.join(self.root, self.images_filename)
        labels_path = os.path.join(self.root, self.labels_filename)
        return (check_integrity(data_path, self.images_md5) and
                check_integrity(labels_path, self.labels_md5))

    def download(self):
        if self._check_integrity():
            print('Files already downloaded and verified')
            return

        root = self.root
        try:
            download_url(self.images_url, root, self.images_filename, self.images_md5)
        except:
            pass
        try:
            download_url(self.labels_url, root, self.labels_filename, self.labels_md5)

        except:
            pass

    def __repr__(self):
        fmt_str = 'Dataset ' + self.__class__.__name__ + '\n'
        fmt_str += '    Number of datapoints: {}\n'.format(self.__len__())
        fmt_str += '    Root Location: {}\n'.format(self.root)
        tmp = '    Transforms (if any): '
        fmt_str += '{0}{1}\n'.format(
            tmp,
            self.transform.__repr__().replace('\n', '\n' + ' ' * len(tmp)))
        tmp = '    Target Transforms (if any): '
        fmt_str += '{0}{1}'.format(
            tmp,
            self.target_transform.__repr__().replace('\n',
                                                     '\n' + ' ' * len(tmp)))
        return fmt_str


def get_cifar10(root=None, val_size=0, seed=None, augment=True, num_workers=4, batch_size=128):
    """ Returns -> [List[CIFAR10], List[CIFAR10]] """
    def train_test_split(dataset, val_size=.1, seed=None):
        N = len(dataset)
        N_test = int(val_size * N)
        N -= N_test

        if seed is not None:
            train, test = random_split(dataset, [N, N_test],
                                       generator=torch.Generator().manual_seed(seed))
        else:
            train, test = random_split(dataset, [N, N_test])

        return train, test

    _CIFAR_TRAIN_TRANSFORM = transforms.Compose([
        transforms.RandomCrop(32, padding=4),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
    ])

    train_data = CIFAR10(root=root, train=True, download=True,
                         transform=_CIFAR_TRAIN_TRANSFORM if augment else _CIFAR_TEST_TRANSFORM)

    test_data = CIFAR10(root=root, train=False, download=True,
                        transform=_CIFAR_TEST_TRANSFORM)

    if val_size != 0:
        train_data, val_data = train_test_split(train_data, val_size=val_size, seed=seed)
        return train_data, val_data, test_data

    train_loader = DataLoader(train_data, batch_size=batch_size, num_workers=num_workers,
                              shuffle=True)
    test_loader = DataLoader(test_data, batch_size=batch_size, num_workers=num_workers)
    N = len(train_data)
    return train_loader, test_loader, N


def load_cifar10_1(root: str) -> CIFAR10_1:
    """Load cifar10.1 dataset from root"""
    val_c_dir = Path('cifar10_1')
    test_data_cifar10_1 = CIFAR10_1(root / val_c_dir, download=True, transform=_CIFAR_TEST_TRANSFORM)
    return test_data_cifar10_1
