import torch
import math
import numpy as np


def bitonic_network(n):
    IDENTITY_MAP_FACTOR = .5
    num_blocks = math.ceil(np.log2(n))
    assert n <= 2 ** num_blocks
    network = []

    for block_idx in range(num_blocks):
        for layer_idx in range(block_idx + 1):
            m = 2 ** (block_idx - layer_idx)

            split_a, split_b = np.zeros((n, 2**num_blocks)), np.zeros((n, 2**num_blocks))
            combine_min, combine_max = np.zeros((2**num_blocks, n)), np.zeros((2**num_blocks, n))
            count = 0

            for i in range(0, 2**num_blocks, 2*m):
                for j in range(m):
                    ix = i + j
                    a, b = ix, ix + m

                    # Cases to handle n \neq 2^k: The top wires are discarded and if a comparator considers them, the
                    # comparator is ignored.
                    if a >= 2**num_blocks-n and b >= 2**num_blocks-n:
                        split_a[count, a], split_b[count, b] = 1, 1
                        if (ix // 2**(block_idx + 1)) % 2 == 1:
                            a, b = b, a
                        combine_min[a, count], combine_max[b, count] = 1, 1
                        count += 1
                    elif a < 2**num_blocks-n and b < 2**num_blocks-n:
                        pass
                    elif a >= 2**num_blocks-n and b < 2**num_blocks-n:
                        split_a[count, a], split_b[count, a] = 1, 1
                        combine_min[a, count], combine_max[a, count] = IDENTITY_MAP_FACTOR, IDENTITY_MAP_FACTOR
                        count += 1
                    elif a < 2**num_blocks-n and b >= 2**num_blocks-n:
                        split_a[count, b], split_b[count, b] = 1, 1
                        combine_min[b, count], combine_max[b, count] = IDENTITY_MAP_FACTOR, IDENTITY_MAP_FACTOR
                        count += 1
                    else:
                        assert False

            split_a = split_a[:count, 2 ** num_blocks - n:]
            split_b = split_b[:count, 2 ** num_blocks - n:]
            combine_min = combine_min[2**num_blocks-n:, :count]
            combine_max = combine_max[2**num_blocks-n:, :count]
            network.append((split_a, split_b, combine_min, combine_max))

    return network


def odd_even_network(n):
    layers = n

    network = []

    shifted: bool = False
    even: bool = n % 2 == 0

    for layer in range(layers):

        if even:
            k = n // 2 + shifted
        else:
            k = n // 2 + 1

        split_a, split_b = np.zeros((k, n)), np.zeros((k, n))
        combine_min, combine_max = np.zeros((n, k)), np.zeros((n, k))

        count = 0

        # for i in range(n // 2 if not (even and shifted) else n // 2 - 1):
        for i in range(int(shifted), n-1, 2):
            a, b = i, i + 1
            split_a[count, a], split_b[count, b] = 1, 1
            combine_min[a, count], combine_max[b, count] = 1, 1
            count += 1

        if even and shifted:
            # Make sure that the corner values stay where they are/were:
            a, b = 0, 0
            split_a[count, a], split_b[count, b] = 1, 1
            combine_min[a, count], combine_max[b, count] = .5, .5
            count += 1
            a, b = n - 1, n - 1
            split_a[count, a], split_b[count, b] = 1, 1
            combine_min[a, count], combine_max[b, count] = .5, .5
            count += 1

        elif not even:
            if shifted:
                a, b = 0, 0
            else:
                a, b = n - 1, n - 1
            split_a[count, a], split_b[count, b] = 1, 1
            combine_min[a, count], combine_max[b, count] = .5, .5
            count += 1

        assert count == k

        network.append((split_a, split_b, combine_min, combine_max))
        shifted = not shifted

    return network


def get_sorting_network(type, n, device):
    def matrix_to_torch(m):
        return [[torch.from_numpy(matrix).float().to(device) for matrix in matrix_set] for matrix_set in m]

    if type == 'bitonic':
        return matrix_to_torch(bitonic_network(n))
    elif type == 'odd_even':
        return matrix_to_torch(odd_even_network(n))
    else:
        raise NotImplementedError('Sorting network `{}` unknown.'.format(type))

