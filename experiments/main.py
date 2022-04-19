# Copyright (c) Felix Petersen.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import argparse
import diffsort
from tqdm import tqdm
import random
import torch
from datasets.dataset import MultiDigitSplits
import models
import utils


def ranking_accuracy(data, targets):
    scores = model(data).squeeze(2)

    acc = torch.argsort(targets, dim=-1) == torch.argsort(scores, dim=-1)

    acc_em = acc.all(-1).float().mean()
    acc_ew = acc.float().mean()

    # EM5:
    scores = scores[:, :5]
    targets = targets[:, :5]
    acc = torch.argsort(targets, dim=-1) == torch.argsort(scores, dim=-1)
    acc_em5 = acc.all(-1).float().mean()

    return dict(
        acc_em=acc_em.type(torch.float32).mean().item(),
        acc_ew=acc_ew.type(torch.float32).mean().item(),
        acc_em5=acc_em5.type(torch.float32).mean().item(),
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='MNIST sorting benchmark')
    parser.add_argument('-b', '--batch_size', type=int, default=100)
    parser.add_argument('-n', '--num_compare', type=int, default=5)
    parser.add_argument('-i', '--num_steps', type=int, default=100_000, help='number of training steps')
    parser.add_argument('-e', '--eval_freq', type=int, default=1_000, help='the evaluation frequency')
    parser.add_argument('-m', '--method', type=str, default='odd_even', choices=['odd_even', 'bitonic'])
    parser.add_argument('-x', '--distribution', type=str, default='cauchy', choices=[
        'cauchy',
        'reciprocal',
        'optimal',
        'gaussian',
        'logistic',
        'logistic_phi',
    ])
    parser.add_argument('-s', '--steepness', type=float, default=10)
    parser.add_argument('-a', '--art_lambda', type=float, default=0.25)
    parser.add_argument('-d', '--dataset', type=str, default='mnist', choices=['mnist', 'svhn'])
    parser.add_argument('-l', '--nloglr', type=float, default=3.5, help='Negative log learning rate')
    parser.add_argument('--device', type=str, default='cuda')
    parser.add_argument('--seed', type=int, default=0)
    args = parser.parse_args()

    random.seed(args.seed)
    torch.manual_seed(args.seed)

    best_valid_acc = 0.

    # ---

    if args.device == 'cuda' and not torch.cuda.is_available():
        print('----------------------------------------------------')
        print('--- WARNING: No GPU detected, running on CPU ... ---')
        print('----------------------------------------------------')
        args.device = 'cpu'

    splits = MultiDigitSplits(dataset=args.dataset, num_compare=args.num_compare, seed=args.seed)

    # drop_last needs to be true, otherwise error with testing for SVHN
    loader_kwargs = dict(batch_size=args.batch_size, drop_last=True)

    if args.dataset == 'svhn':
        loader_kwargs['batch_size'] = args.batch_size * args.num_compare

        def collate_fn(batch):
            data, targets = zip(*batch)
            data = torch.stack(data)
            targets = torch.tensor(targets)  # targets is a tuple of int
            data = data.reshape(args.batch_size, args.num_compare, *data.shape[1:])
            targets = targets.reshape(args.batch_size, args.num_compare, *targets.shape[1:])
            return data, targets

        loader_kwargs['collate_fn'] = collate_fn

    data_loader_train = splits.get_train_loader(**loader_kwargs)
    data_loader_valid = splits.get_valid_loader(**loader_kwargs)
    data_loader_test = splits.get_test_loader(**loader_kwargs)

    if args.dataset == 'mnist':
        model = models.MultiDigitMNISTNet().to(args.device)
    elif args.dataset == 'svhn':
        model = models.SVHNConvNet().to(args.device)
    else:
        raise ValueError(args.dataset)

    optim = torch.optim.Adam(model.parameters(), lr=10**(-args.nloglr))

    sorter = diffsort.DiffSortNet(
        sorting_network_type=args.method,
        size=args.num_compare,
        device=args.device,
        steepness=args.steepness,
        art_lambda=args.art_lambda,
    )

    valid_accs = []
    test_acc = None

    for iter_idx, (data, targets) in tqdm(
        enumerate(utils.load_n(data_loader_train, args.num_steps)),
        desc="Training steps",
        total=args.num_steps,
    ):
        data = data.to(args.device)
        targets = targets.to(args.device)

        outputs = model(data).squeeze(2)
        _, perm_prediction = sorter(outputs)

        perm_ground_truth = torch.nn.functional.one_hot(torch.argsort(targets, dim=-1)).transpose(-2, -1).float()
        loss = torch.nn.BCELoss()(perm_prediction, perm_ground_truth)

        optim.zero_grad()
        loss.backward()
        optim.step()

        if (iter_idx + 1) % args.eval_freq == 0:

            current_valid_accs = []
            for data, targets in data_loader_valid:
                data, targets = data.to(args.device), targets.to(args.device)
                current_valid_accs.append(ranking_accuracy(data, targets))
            valid_accs.append(utils.avg_list_of_dicts(current_valid_accs))

            print(iter_idx, 'valid', valid_accs[-1])

            if valid_accs[-1]['acc_em5'] > best_valid_acc:
                best_valid_acc = valid_accs[-1]['acc_em5']

                current_test_accs = []
                for data, targets in data_loader_test:
                    data, targets = data.to(args.device), targets.to(args.device)
                    current_test_accs.append(ranking_accuracy(data, targets))
                test_acc = utils.avg_list_of_dicts(current_test_accs)

                print(iter_idx, 'test', test_acc)

    print('final test', test_acc)
