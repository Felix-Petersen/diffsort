# Copyright (c) Felix Petersen.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import collections
import functools
import operator


def avg_list_of_dicts(list_of_dicts):
    summed = functools.reduce(operator.add, map(collections.Counter, list_of_dicts))
    averaged = {k: summed[k] / len(list_of_dicts) for k in summed}
    return averaged


def load_n(loader, n):
    i = 0
    while i < n:
        for x in loader:
            yield x
            i += 1
            if i == n:
                break
