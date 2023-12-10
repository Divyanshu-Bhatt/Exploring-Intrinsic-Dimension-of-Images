import torch
from utils import getKnearestNeighbours


class TwoNN(object):
    """
    Find the intrinsic dimension of a dataset using TwoNN.
    """

    def __init__(self, discard_fraction=0.1, batch_size=None, device="cpu"):
        """
        Parameters
        ----------
        discard_fraction : float
            The fraction of points to discard. (0 < discard_fraction < 1)
        batch_size : int, optional (default=None)
            The number of samples to process at a time. If None, all samples are processed at once.
        device : str (default='cpu')
            The device to use for the computation. (cpu, cuda, mps)
        """

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
        nearest_two_distances = getKnearestNeighbours(
            2, X, self.batch_size, distance=True
        )

        _mu = nearest_two_distances[:, 1] / nearest_two_distances[:, 0]
        mu = _mu[torch.argsort(_mu)][: int(len(X) * (1 - self.discard_fraction))]

        Femp = torch.arange(len(mu)) / len(X)

        x_axis = torch.log(mu).reshape(-1, 1).float().to(self.device)
        y_axis = -torch.log(1 - Femp).reshape(-1, 1).to(self.device)

        slope = torch.linalg.lstsq(x_axis, y_axis)[0]

        return slope.item()
