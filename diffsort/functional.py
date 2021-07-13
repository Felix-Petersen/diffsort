import torch
from typing import List, Tuple

SORTING_NETWORK_TYPE = List[torch.tensor]


def execute_sort(
        sorting_network,
        vectors,
        steepness=10.,
        art_lambda=0.25,
        softmax_fn='logistic_phi'
):
    x = vectors
    X = torch.eye(vectors.shape[1]).to(sorting_network[0][0].device).repeat(vectors.shape[0], 1, 1)

    for split_a, split_b, combine_min, combine_max in sorting_network:
        a, b = x @ split_a.T, x @ split_b.T

        if softmax_fn == 'logistic':
            alpha = torch.sigmoid((b-a) * steepness)

        elif softmax_fn == 'logistic_phi':
            alpha = torch.sigmoid((b-a) * steepness / ((a-b).abs() + 1.e-10).pow(art_lambda))

        else:
            raise NotImplementedError('softmax method `{}` unknown'.format(softmax_fn))

        aX = X @ split_a.T
        bX = X @ split_b.T
        w_min = alpha.unsqueeze(-2) * aX + (1-alpha).unsqueeze(-2) * bX
        w_max = (1-alpha).unsqueeze(-2) * aX + alpha.unsqueeze(-2) * bX
        X = (w_max @ combine_max.T.unsqueeze(-3)) + (w_min @ combine_min.T.unsqueeze(-3))
        x = (alpha * a + (1-alpha) * b) @ combine_min.T + ((1-alpha) * a + alpha * b) @ combine_max.T
    return x, X


def sort(
        sorting_network: SORTING_NETWORK_TYPE,
        vectors: torch.Tensor,
        steepness: float = 10.0,
        art_lambda: float = 0.25,
        softmax_fn: str = "logistic_phi"
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Sort a matrix along axis 1 using differentiable sorting networks. Return the permutation matrix.

    Positional arguments:
    sorting_network
    vectors -- the matrix to sort along axis 1; sorted in-place

    Keyword arguments:
    steepness -- relevant for sigmoid and sigmoid_phi interpolation (default 10.0)
    art_lambda -- relevant for logistic_phi interpolation (default 0.25)
    interpolation_type -- how to interpolate when swapping two numbers; supported: `logistic`, `logistic_phi`,
    `discrete` (default 'logistic_phi')
    """
    assert sorting_network[0][0].device == vectors.device, (
        f"The sorting network is on device {sorting_network[0][0].device} while the vectors are on device"
        f" {vectors.device}, but they both need to be on the same device."
    )
    return execute_sort(
        sorting_network=sorting_network,
        vectors=vectors,
        steepness=steepness,
        art_lambda=art_lambda,
        softmax_fn=softmax_fn
    )