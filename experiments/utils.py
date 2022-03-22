# Copyright (c) Felix Petersen.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import collections
import functools
import operator
import numpy as np


def avg_list_of_dicts(list_of_dicts):
    result = {}
    for k in list_of_dicts[0]:
        result[k] = np.mean([d[k] for d in list_of_dicts])
    return result


def load_n(loader, n):
    i = 0
    while i < n:
        for x in loader:
            yield x
            i += 1
            if i == n:
                break
