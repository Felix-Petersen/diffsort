# diffsort - Differentiable Sorting Networks

![diffsort_logo](diffsort_logo.png)

Official implementation for our ICML 2021 Paper "Differentiable Sorting Networks for Scalable Sorting and Ranking Supervision".
In this work, we leverage classic sorting networks and relax them to propose a new differentiable sorting function: diffsort.
Diffsort allows propagating gradients through a smooth differentiable approximate sorting function.
Herein, diffsort outperforms existing differentiable sorting functions on the four-digit MNIST and the SVHN sorting tasks.
In this repo, we present the PyTorch implementation of our ICML 2021 paper on differentiable sorting networks.
Paper @ [ArXiv](https://arxiv.org/pdf/2105.04019.pdf),
Video @ [Youtube](https://www.youtube.com/watch?v=38dvqdYEs1o).

## Installation

`diffsort` can be installed via pip from PyPI with
```bash
pip install diffsort
```

Or from source, e.g., in a virtual environment like
```bash
virtualenv -p python3 .env1
. .env1/bin/activate
pip install .
```

## Usage

```python
import torch
from diffsort import DiffSortNet

vector_length = 2**4
vectors = torch.randperm(vector_length, dtype=torch.float32, device='cpu', requires_grad=True).view(1, -1)
vectors = vectors - 5.

# sort using a bitonic-sorting-network
sorter = DiffSortNet('bitonic', vector_length, steepness=5)
sorted_vectors, permutation_matrices = sorter(vectors)
print(sorted_vectors)
```

## Experiments

Will be published soon.

## Citing

```bibtex
@inproceedings{Petersen2021-diffsort,
  title={Differentiable Sorting Networks for Scalable Sorting and Ranking Supervision},
  author={Petersen, Felix and Borgelt, Christian and Kuehne, Hilde and Deussen, Oliver},
  journal={International Conference on Machine Learning (ICML)},
  year={2021}
}
```

## License

`diffsort` is released under the MIT license. See [LICENSE](LICENSE) for additional details about it.

