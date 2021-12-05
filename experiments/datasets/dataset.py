# Copyright (c) Felix Petersen.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import torch
from torchvision import datasets
from torch.utils.data import Dataset, DataLoader
from .svhn_multi import SVHNMultiDigit


class MultiDigitDataset(Dataset):

    def __init__(
            self,
            images,
            labels,
            num_digits,
            num_compare,
            seed=0,
            determinism=True,
    ):
        super(MultiDigitDataset, self).__init__()
        self.images = images
        self.labels = labels
        self.num_digits = num_digits
        self.num_compare = num_compare
        self.seed = seed
        self.rand_state = None

        self.determinism = determinism

        if determinism:
            self.reset_rand_state()

    def __len__(self):
        return self.images.shape[0]

    def __getitem__(self, idx):

        if self.determinism:
            prev_state = torch.random.get_rng_state()
            torch.random.set_rng_state(self.rand_state)

        labels = []
        images = []
        labels_ = None
        for digit_idx in range(self.num_digits):
            id = torch.randint(len(self), (self.num_compare, ))
            labels.append(self.labels[id])
            images.append(self.images[id].type(torch.float32) / 255.)
            if labels_ is None:
                labels_ = torch.zeros_like(labels[0] * 1.)
            labels_ = labels_ + 10.**(self.num_digits - 1 - digit_idx) * self.labels[id]

        images = torch.cat(images, dim=-1)

        if self.determinism:
            self.rand_state = torch.random.get_rng_state()
            torch.random.set_rng_state(prev_state)

        return images, labels_

    def reset_rand_state(self):
        prev_state = torch.random.get_rng_state()
        torch.random.manual_seed(self.seed)
        self.rand_state = torch.random.get_rng_state()
        torch.random.set_rng_state(prev_state)


class MultiDigitSplits(object):
    def __init__(self, dataset, num_digits=4, num_compare=None, seed=0, deterministic_data_loader=True):

        self.deterministic_data_loader = deterministic_data_loader

        if dataset == 'mnist':
            trva_real = datasets.MNIST(root='./data-mnist', download=True)
            xtr_real = trva_real.data[:55000].view(-1, 1, 28, 28)
            ytr_real = trva_real.targets[:55000]
            xva_real = trva_real.data[55000:].view(-1, 1, 28, 28)
            yva_real = trva_real.targets[55000:]

            te_real = datasets.MNIST(root='./data-mnist', train=False, download=True)
            xte_real = te_real.data.view(-1, 1, 28, 28)
            yte_real = te_real.targets

            self.train_dataset = MultiDigitDataset(
                images=xtr_real, labels=ytr_real, num_digits=num_digits, num_compare=num_compare, seed=seed,
                determinism=deterministic_data_loader)
            self.valid_dataset = MultiDigitDataset(
                images=xva_real, labels=yva_real, num_digits=num_digits, num_compare=num_compare, seed=seed)
            self.test_dataset = MultiDigitDataset(
                images=xte_real, labels=yte_real, num_digits=num_digits, num_compare=num_compare, seed=seed)

        elif dataset == 'svhn':
            self.train_dataset = SVHNMultiDigit(root='./data-svhn', split='train', download=True)
            self.valid_dataset = SVHNMultiDigit(root='./data-svhn', split='val', download=True)
            self.test_dataset = SVHNMultiDigit(root='./data-svhn', split='test', download=True)
        else:
            raise NotImplementedError()

    def get_train_loader(self, batch_size, **kwargs):
        train_loader = DataLoader(self.train_dataset,
                                  batch_size=batch_size,
                                  num_workers=4 if not self.deterministic_data_loader else 0,
                                  shuffle=True, **kwargs)
        return train_loader

    def get_valid_loader(self, batch_size, **kwargs):
        valid_loader = DataLoader(self.valid_dataset,
                                  batch_size=batch_size, shuffle=False, **kwargs)
        return valid_loader

    def get_test_loader(self, batch_size, **kwargs):
        test_loader = DataLoader(self.test_dataset,
                                 batch_size=batch_size, shuffle=False, **kwargs)
        return test_loader
