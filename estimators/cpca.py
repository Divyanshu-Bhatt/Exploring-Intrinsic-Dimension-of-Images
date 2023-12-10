import torch
from utils import getKnearestNeighbours, CoverSet
from tqdm import tqdm


class CPCA(object):
    """
    Find the intrinsic dimension of a dataset using PCA with cover sets.
    """

    def __init__(
        self,
        k,
        alpha=10,
        beta=0.8,
        condition="max_variance",
        batch_size=None,
        device="cpu",
    ):
        """
        Parameters
        ----------
        k : int
            The number of neighbours to find.
        alpha : float
            The alpha parameter (large gap threshold). (alpha >> 1)
        beta : float
            The percentage of total covariance in non-noise. (0 < beta < 1)
        condition : str (default='max_variance')
            The condition to use for the estimator. (max_variance or sum_variance)
        batch_size : int, optional (default=None)
            The number of samples to process at a time. If None, all samples are processed at once.
        device : str (default='cpu')
            The device to use for the computation. (cpu, cuda, mps)
        """

        self.k = k
        self.alpha = alpha
        self.beta = beta
        self.device = device
        self.condition = condition
        self.batch_size = batch_size

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
        self.coverset = self._getCoverSet(X)
        return self._getDimensions(X)

    def _getCoverSet(self, X):
        """
        Find minimum cover set for X.

        Parameters
        ----------
        X : torch.Tensor, shape (n_samples, n_features)
            The training input samples.

        Returns
        -------
        cover_set : CoverSet
            The cover set for X.
        """

        indices = getKnearestNeighbours(self.k, X, self.batch_size)
        return CoverSet(indices, X, device=self.device)

    def _getDimensions(self, X):
        """
        Find the intrinsic dimension using the coverset.

        Parameters
        ----------
        X : torch.Tensor, shape (n_samples, n_features)
            The training input samples.

        Returns
        -------
        dimension : int
            The intrinsic dimension of the data.
        """

        eigenvalues = []
        for i in tqdm(range(len(self.coverset.F)), desc="Finding eigenvalues"):
            cover_set = X[self.coverset.F[i]]
            covariance_matrix = torch.cov(cover_set.T)
            eigenvalues_ = torch.linalg.eigvals(covariance_matrix)
            eigenvalues_ = torch.real(eigenvalues_)
            eigenvalues.append(eigenvalues_)

        eigenvalues = torch.stack(eigenvalues, dim=0)
        eigenvalues = torch.sum(eigenvalues, dim=0)
        eigenvalues = torch.sort(eigenvalues, descending=True)[0]

        if self.condition == "max_variance":
            dimension_ = self._fitMaxVariance(eigenvalues)
        elif self.condition == "sum_variance":
            dimension_ = self._fitSumVariance(eigenvalues)

        return dimension_

    def _fitMaxVariance(self, eigenvalues):
        """
        Find the maximum value of dimension "d" which satisfies min(eigenvalues[:d]) > alpha * max(eigenvalues[d:])

        Parameters
        ----------
        eigenvalues : torch.Tensor, shape (n_features,)
            The eigenvalues of the covariance matrix of the data in decreasing order.

        Returns
        -------
        d : int
            The intrinsic dimension of the data.
        """

        d = 1
        while (d < len(eigenvalues)) and (
            eigenvalues[d - 1] < self.alpha * eigenvalues[d]
        ):
            d += 1

        return d - 1

    def _fitSumVariance(self, eigenvalues):
        """
        Find the maximum value of dimension "d" which satisfies sum(eigenvalues[:d]) > beta * sum(eigenvalues)

        Parameters
        ----------
        eigenvalues : torch.Tensor, shape (n_features,)
            The eigenvalues of the covariance matrix of the data in decreasing order.

        Returns
        -------
        d : int
            The intrinsic dimension of the data.
        """

        d = 1
        sum_eigenvalues = torch.sum(eigenvalues)
        sum_eigenvalues_d = torch.sum(eigenvalues[:d])

        while (d < len(eigenvalues)) and (
            sum_eigenvalues_d < self.beta * sum_eigenvalues
        ):
            sum_eigenvalues_d += eigenvalues[d]
            d += 1

        return d - 1
