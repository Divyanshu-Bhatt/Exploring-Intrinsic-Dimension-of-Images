# import torch
# from utils import getKnearestNeighbours


# class Gride(object):
#     """
#     Find the intrinsic dimension of a dataset using Gride.
#     """

#     def __init__(
#         self, k1=10, k2=20, discard_fraction=0.1, batch_size=None, device="cpu"
#     ):
#         """
#         Parameters
#         ----------
#         k1 : int
#             The closer nearest neighbour to use.
#         k2 : int
#             The farther nearest neighbour to use.
#         discard_fraction : float
#             The fraction of points to discard. (0 < discard_fraction < 1)
#         batch_size : int, optional (default=None)
#             The number of samples to process at a time. If None, all samples are processed at once.
#         device : str (default='cpu')
#             The device to use for the computation. (cpu, cuda, mps)
#         """

#         self.k1 = k1
#         self.k2 = k2
#         self.discard_fraction = discard_fraction
#         self.batch_size = batch_size
#         self.device = device

#     def fit(self, X):
#         """
#         Fit the estimator to the data in X.

#         Parameters
#         ----------
#         X : torch.Tensor, shape (n_samples, n_features)
#             The training input samples.

#         Returns
#         -------
#         dimension_ : int
#             The intrinsic dimension of the data.
#         """

#         X = X.to(self.device)
#         nearest_neighbours = getKnearestNeighbours(
#             self.k2, X, self.batch_size, distance=True
#         )

#         nearest_neighbours = torch.cat(
#             [
#                 nearest_neighbours[:, self.k1 - 1].reshape(-1, 1),
#                 nearest_neighbours[:, self.k2 - 1].reshape(-1, 1),
#             ],
#             dim=1,
#         )

#         _mu = nearest_neighbours[:, 1] / nearest_neighbours[:, 0]
#         mu = _mu[torch.argsort(_mu)][: int(len(X) * (1 - self.discard_fraction))]

#         Femp = torch.arange(len(mu)) / len(X)

#         x_axis = torch.log(mu).reshape(-1, 1).float().to(self.device)
#         y_axis = -torch.log(1 - Femp).reshape(-1, 1).to(self.device)

#         slope = torch.linalg.lstsq(x_axis, y_axis)[0]

#         return slope.item()
import torch
import cvxpy as cp
from utils import getKnearestNeighbours


class Gride(object):
    """
    Find the intrinsic dimension of a dataset using Gride.
    """

    def __init__(self, k=10, discard_fraction=0.1, batch_size=None, device="cpu"):
        """
        Parameters
        ----------
        k1 : int
            The closer nearest neighbour to use.
        k2 : int
            The farther nearest neighbour to use.
        discard_fraction : float
            The fraction of points to discard. (0 < discard_fraction < 1)
        batch_size : int, optional (default=None)
            The number of samples to process at a time. If None, all samples are processed at once.
        device : str (default='cpu')
            The device to use for the computation. (cpu, cuda, mps)
        """

        self.k1 = k
        self.discard_fraction = discard_fraction
        self.batch_size = batch_size
        self.device = device

    def fit(self, X):
        """
        Fit the estimator to the data in X.

        Parameters
        ----------
        X : torch.Tensor, shape (n_samples, n_features)
            The training input samples.

        Returns
        -------
        dimension_ : int
            The intrinsic dimension of the data.
        """

        X = X.to(self.device)
        nearest_neighbours = getKnearestNeighbours(
            self.k2, X, self.batch_size, distance=True
        )

        nearest_neighbours = torch.cat(
            [
                nearest_neighbours[:, self.k1 - 1].reshape(-1, 1),
                nearest_neighbours[:, self.k2 - 1].reshape(-1, 1),
            ],
            dim=1,
        )

        _mu = nearest_neighbours[:, 1] / nearest_neighbours[:, 0]
        mu = _mu[torch.argsort(_mu)][: int(len(X) * (1 - self.discard_fraction))]

        N = len(mu)
