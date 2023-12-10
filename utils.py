import torch
from tqdm import tqdm


class CoverSet(object):
    """
    Cover Set Data Structure.
    """

    def __init__(self, k_neareast_neighbours, X, device="cpu"):
        """
        Parameters
        ----------
        k_neareast_neighbours : (n_samples, k) torch.Tensor
            The indices of the k nearest neighbours of each point in the data.
        X : torch.Tensor, shape (n_samples, n_features)
            The training input samples.
        device : str, optional (default="cpu")
            The device to use for the computation. (cpu, cuda, mps)
        """

        self.F = k_neareast_neighbours.to(device)
        self.q = torch.zeros(len(k_neareast_neighbours)).to(device)
        self.r = torch.zeros(len(k_neareast_neighbours)).to(device)

        self.computeQ()
        self.update(X)

    def computeQ(self):
        """
        Compute the frequency of xi in the k nearest neighbours
        """

        for i in range(len(self.F)):
            self.q[self.F[i]] += 1

    def update(self, X):
        """
        Create the Cover set dataset given the indices of the k nearest neighbours of each point in the data.

        Parameters
        ----------
        X : torch.Tensor, shape (n_samples, n_features)
            The training input samples.
        """

        for i in tqdm(range(len(self.F)), desc="Updating Cover Set"):
            if torch.all(self.q[self.F[i]] > 1):
                self.q[self.F] -= 1
            else:
                self.r[i] = self.getMaxDistance(X[self.F[i]])

        indices = torch.where(self.r != 0)
        self.r = self.r[indices]
        self.q = self.q[indices]
        self.F = self.F[indices]

    def getMaxDistance(self, X):
        """
        Finds the maximum distance possible between the points provided

        Parameters
        ----------
        X : torch.Tensor, shape (n_samples, n_features)
            The training input samples.
        """

        distances = torch.cdist(X, X, p=2)
        return torch.max(distances)


def getKnearestNeighbours(k, data, batch_size=None, distance=False):
    """
    Get the indices of the k nearest neighbours of each point in the data.

    Parameters
    ----------
    k : int
        The number of neighbours to find.
    data : torch.Tensor, shape (n_samples, n_features)
        The data.
    batch_size : int, optional (default=None)
        The number of samples to process at a time. If None, all samples are processed at once.
    distance : bool, optional (default=False)
        True if we want to return the distances instead of the indices.

    Returns
    -------
    indices : torch.Tensor, shape (n_samples, k)
        The indices of the k nearest neighbours of each point in the data.
                                  OR
    distances : torch.Tensor, shape (n_samples, k)
        The distances of the k nearest neighbours of each point in the data.
    """

    if batch_size is None:
        batched_data = [data]
    else:
        batched_data = torch.split(data, batch_size)
    indices, distances = [], []

    for batch in tqdm(batched_data, desc="Finding Nearest Neighbour Batch"):
        dist = torch.cdist(
            batch, data, p=2, compute_mode="donot_use_mm_for_euclid_dist"
        )
        batch_distances, batch_indices = torch.topk(dist, k=k + 1, dim=1, largest=False)

        # Removing the first column as it is the distance of the point to itself
        batch_indices = batch_indices[:, 1:]
        batch_distances = batch_distances[:, 1:]

        if not distance:
            indices.append(batch_indices)
        else:
            distances.append(batch_distances)

    if not distance:
        indices = torch.cat(indices, dim=0)
        return indices
    else:
        distances = torch.cat(distances, dim=0)
        return distances
