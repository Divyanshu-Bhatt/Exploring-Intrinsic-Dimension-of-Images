import torch


class PCA(object):
    """
    Find the intrinsic dimension of a dataset using PCA.
    """

    def __init__(self, alpha=10, beta=0.8, condition="max_variance", device="cpu"):
        """
        Parameters
        ----------
        alpha : float
            The alpha parameter (large gap threshold). (alpha >> 1)
        beta : float
            The percentage of total covariance in non-noise. (0 < beta < 1)
        condition : str (default='max_variance')
            The condition to use for the estimator. (max_variance or sum_variance)
        device : str (default='cpu')
            The device to use for the computation. (cpu, cuda, mps)
        """

        self.alpha = alpha
        self.beta = beta
        self.device = device
        self.condition = condition

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

        covariance_matrix = torch.cov(X.T)
        eigenvalues = torch.linalg.eigvals(covariance_matrix)
        eigenvalues = torch.real(eigenvalues)
        eigenvalues = torch.sort(eigenvalues, descending=True).values

        if self.condition == "max_variance":
            self.dimension_ = self._fitMaxVariance(eigenvalues)
        elif self.condition == "sum_variance":
            self.dimension_ = self._fitSumVariance(eigenvalues)

        return self.dimension_

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
