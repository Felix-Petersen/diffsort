# Copyright (c) Felix Petersen.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

from torchvision.datasets.vision import VisionDataset
from PIL import Image
import os
import os.path
import numpy as np
from torchvision.datasets.utils import download_url, check_integrity, verify_str_arg
import torch
import torchvision.transforms as transforms


class SVHNMultiDigit(VisionDataset):
    """`Preprocessed SVHN-Multi <>`_ Dataset.
    Note: The preprocessed SVHN dataset is based on the the `Format 1` official dataset.
    By cropping the numbers from the images, adding a margin of :math:`30\%` , and resizing to :math:`64\times64` ,
    the dataset has been preprocessed.
    The data split is as follows:

        * ``train``: (30402 of 33402 original ``train``) + (200353 of 202353 original ``extra``)
        * ``val``: (3000 of 33402 original ``train``) + (2000 of 202353 original ``extra``)
        * ``test``: (all of 13068 original ``test``)

    Each ```train / val`` split has been performed using
    ``sklearn.model_selection import train_test_split(data_X_y_tuples, test_size=3000 / 2000, random_state=0)`` .
    This is the closest that we could come to the
    `work by Goodfellow et al. 2013 <https://arxiv.org/pdf/1312.6082.pdf>`_ .

    Args:
        root (string): Root directory of dataset where directory
            ``SVHNMultiDigit`` exists.
        split (string): One of {'train', 'val', 'test'}.
            Accordingly dataset is selected.
        transform (callable, optional): A function/transform that  takes in an PIL image
            and returns a transformed version. E.g, ``transforms.RandomCrop`` .
            (default = random 54x54 crop + normalization)
        target_transform (callable, optional): A function/transform that takes in the
            target and transforms it.
        download (bool, optional): If true, downloads the dataset from the internet and
            puts it in root directory. If dataset is already downloaded, it is not
            downloaded again.
    """

    split_list = {
        'train': ["https://nyc3.digitaloceanspaces.com/publicdata1/svhn-multi-digit-3x64x64_train.p",
                  "svhn-multi-digit-3x64x64_train.p", "25df8732e1f16fef945c3d9a47c99c1a"],
        'val': ["https://nyc3.digitaloceanspaces.com/publicdata1/svhn-multi-digit-3x64x64_val.p",
                "svhn-multi-digit-3x64x64_val.p", "fe5a3b450ce09481b68d7505d00715b3"],
        'test': ["https://nyc3.digitaloceanspaces.com/publicdata1/svhn-multi-digit-3x64x64_test.p",
                 "svhn-multi-digit-3x64x64_test.p", "332977317a21e9f1f5afe7ef47729c5c"]
    }

    def __init__(self, root, split='train',
                 transform=transforms.Compose([
                     transforms.RandomCrop([54, 54]),
                     transforms.ToTensor(),
                     transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])
                 ]),
                 target_transform=None, download=False):
        super(SVHNMultiDigit, self).__init__(root, transform=transform, target_transform=target_transform)
        self.split = verify_str_arg(split, "split", tuple(self.split_list.keys()))
        self.url = self.split_list[split][0]
        self.filename = self.split_list[split][1]
        self.file_md5 = self.split_list[split][2]

        if download:
            self.download()

        if not self._check_integrity():
            raise RuntimeError('Dataset not found or corrupted.' +
                               ' You can use download=True to download it')

        data = torch.load(os.path.join(self.root, self.filename))

        self.data = data[0]
        # loading gives an np array of type np.uint8
        # converting to np.int64, so that we have a LongTensor after
        # the conversion from the numpy array
        # the squeeze is needed to obtain a 1D tensor
        self.labels = data[1].type(torch.LongTensor)

    def __getitem__(self, index):
        """
        Args:
            index (int): Index
        Returns:
            tuple: (image, target) where target is index of the target class.
        """
        img, target = self.data[index], int(self.labels[index])

        # doing this so that it is consistent with all other datasets
        # to return a PIL Image
        img = Image.fromarray(np.transpose(img.numpy(), (1, 2, 0)))

        if self.transform is not None:
            img = self.transform(img)

        if self.target_transform is not None:
            target = self.target_transform(target)

        return img, target

    def __len__(self):
        return len(self.data)

    def _check_integrity(self):
        root = self.root
        md5 = self.split_list[self.split][2]
        fpath = os.path.join(root, self.filename)
        return check_integrity(fpath, md5)

    def download(self):
        md5 = self.split_list[self.split][2]
        download_url(self.url, self.root, self.filename, md5)

    def extra_repr(self):
        return "Split: {split}".format(**self.__dict__)