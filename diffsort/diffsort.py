import torch
from .functional import sort
from .networks import get_sorting_network


class DiffSortNet(torch.nn.Module):
    """Sort a matrix along axis 1 using differentiable sorting networks. Return the permutation matrix.

    Positional arguments:
    sorting_network_type -- which sorting network to use for sorting.
    vectors -- the matrix to sort along axis 1; sorted in-place

    Keyword arguments:
    steepness -- relevant for sigmoid and sigmoid_phi interpolation (default 10.0)
    art_lambda -- relevant for sigmoid_phi interpolation (default 0.25)
    interpolation_type -- how to interpolate when swapping two numbers; supported: `logistic`, `logistic_phi`,
                 (default 'logistic_phi')
    """
    def __init__(
        self,
        sorting_network_type: str,
        size: int,
        device: str = 'cpu',
        steepness: float = 10.0,
        art_lambda: float = 0.25,
        interpolation_type: str = None,
        distribution: str = 'cauchy',
    ):
        super(DiffSortNet, self).__init__()
        self.sorting_network_type = sorting_network_type
        self.size = size

        self.sorting_network = get_sorting_network(sorting_network_type, size, device)

        if interpolation_type is not None:
            assert distribution is None or distribution == 'cauchy' or distribution == interpolation_type, (
                'Two different distributions have been set (distribution={} and interpolation_type={}); however, '
                'they have the same interpretation and interpolation_type is a deprecated argument'.format(
                    distribution, interpolation_type
                )
            )
            distribution = interpolation_type

        self.steepness = steepness
        self.art_lambda = art_lambda
        self.distribution = distribution

    def forward(self, vectors):
        assert len(vectors.shape) == 2
        assert vectors.shape[1] == self.size
        return sort(
            self.sorting_network, vectors, self.steepness, self.art_lambda, self.distribution
        )
