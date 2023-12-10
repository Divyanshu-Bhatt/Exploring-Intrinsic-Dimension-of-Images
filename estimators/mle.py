import torch
from utils import getKnearestNeighbours
from tqdm import tqdm


class MLE(object):
    """
    Find the intrinsic dimension of a dataset using MLE.
    """

    def __init__(self, k1, k2, mackay=False, batch_size=None, device="cpu"):
        """
        Parameters
        ----------
        k1 : int
            The lower limit of the number of nearest neighbours to use.
        k2 : int
            The upper limit of the number of nearest neighbours to use.
        mackay: bool
            Whether to perform the correction pointed out by David MacKay or not.
        batch_size : int, optional (default=None)
            The number of samples to process at a time. If None, all samples are processed at once.
        device : str (default='cpu')
            The device to use for the computation. (cpu, cuda, mps)
        """

        self.k1 = k1
        self.k2 = k2
        self.mackay = mackay
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

        dimesnions_ = []
        distances = getKnearestNeighbours(
            self.k2 + 1, X, self.batch_size, distance=True
        )
        dimesnions_.append(self._fitMLE(distances))

        for k in tqdm(range(self.k1, self.k2 + 1), desc="MLE"):
            distances_ = distances[:, : k + 1]
            dimesnions_.append(self._fitMLE(distances_))

        return torch.mean(torch.tensor(dimesnions_)).item()

    def _fitMLE(self, distances):
        """
        Find the intrinsic dimension using the MLE estimator.

        Parameters
        ----------
        distances : torch.Tensor, shape (n_samples, k_neighbours)
            The distances to the k nearest neighbours of each point in the data.

        Returns
        -------
        dimension : int
            The intrinsic dimension of the data.
        """

        ratio = distances[:, :-1] / distances[:, -1].reshape(-1, 1)
        log_ratio = torch.log(ratio)
        dk_x = -1 / torch.mean(log_ratio, dim=1)

        if self.mackay:
            dk_x_inverse_mean = torch.mean(1 / dk_x)
            return 1 / dk_x_inverse_mean
        else:
            return torch.mean(dk_x)
