import os
import time
import skdim
import torch
import torchvision
import pandas as pd
from estimators.pca import PCA
from estimators.cpca import CPCA
from estimators.mle import MLE
from estimators.twonn import TwoNN
from tqdm import tqdm
import argparse

parser = argparse.ArgumentParser()
parser.add_argument(
    "--dataset",
    type=str,
    default="toy",
    help="Dataset to use (toy, MNIST, CIFAR10, generated)",
)
parser.add_argument(
    "--device",
    type=str,
    default="cpu",
    help="Device to use (cpu, cuda, mps)",
)
parser.add_argument(
    "--batch_size",
    type=int,
    default=1000,
    help="Batch size to use for estimators",
)
parser.add_argument(
    "--latent_dim",
    type=int,
    default=8,
    help="Latent dimension of generated dataset",
)
args = parser.parse_args()

device = args.device
batch_size = args.batch_size
latent_dim = args.latent_dim

if args.dataset == "toy":
    benchmark = skdim.datasets.BenchmarkManifolds(random_state=0)
    dict_data = benchmark.generate(n=10000)
    df = benchmark.truth

    # Empty dataframe to store results
    results = pd.DataFrame(
        columns=["Ground Truth", "PCA", "CPCA", "MLE", "MLE (corrected)", "TwoNN"]
    )

    for idx, keys in enumerate(tqdm(dict_data.keys())):
        data = dict_data[keys]
        data = torch.from_numpy(data).float()

        true_dim = df.iloc[idx, 0]
        pca_dim = PCA(device=device).fit(data)
        cpca_dim = CPCA(k=20, batch_size=1000, device=device).fit(data)
        mle_dim = MLE(k1=12, k2=20, batch_size=1000, device=device).fit(data)
        mle_dim_corrected = MLE(
            k1=12, k2=20, mackay=True, batch_size=1000, device=device
        ).fit(data)
        twonn_dim = TwoNN(batch_size=1000, device=device).fit(data)

        results.loc[keys] = [
            true_dim,
            pca_dim,
            cpca_dim,
            mle_dim,
            mle_dim_corrected,
            twonn_dim,
        ]

    results.to_csv("toy_dataset_results.csv")

elif args.dataset == "generated":
    file_loc = f"samples/basenji_{latent_dim}"
    data = []

    for file in os.listdir(file_loc):
        if file.split(".")[-1] != "pt":
            continue
        data_ = torch.load(os.path.join(file_loc, file))
        data.append(data_)

    data = torch.concat(data, axis=0)
    data = data.reshape(data.shape[0], -1).float()

    print(f"Loaded Basenji {latent_dim} dataset")
    data = data.to(device)
    print("Data shape : ", data.shape)

    # start = time.time()
    # pca_dim = PCA(device=device).fit(data)
    # print("PCA : ", pca_dim)
    # pca_time = time.time() - start

    # start = time.time()
    # cpca_dim = CPCA(k=20, batch_size=batch_size, device=device).fit(data)
    # print("CPCA : ", cpca_dim)
    # cpca_time = time.time() - start

    start = time.time()
    mle_dim = MLE(k1=12, k2=20, batch_size=batch_size, device=device).fit(data)
    print("MLE : ", mle_dim)
    mle_time = time.time() - start

    start = time.time()
    mle_dim_corrected = MLE(
        k1=12, k2=20, mackay=True, batch_size=batch_size, device=device
    ).fit(data)
    print("MLE (corrected) : ", mle_dim_corrected)
    mle_time_corrected = time.time() - start

    start = time.time()
    twonn_dim = TwoNN(batch_size=batch_size, device=device).fit(data)
    print("TwoNN : ", twonn_dim)
    twonn_time = time.time() - start

    # results = pd.DataFrame(columns=["PCA", "CPCA", "MLE", "MLE (corrected)", "TwoNN"])
    results = pd.DataFrame(columns=["MLE", "MLE (corrected)", "TwoNN"])

    results = results.append(
        {
            # "PCA": pca_dim,
            # "CPCA": cpca_dim,
            "MLE": mle_dim,
            "MLE (corrected)": mle_dim_corrected,
            "TwoNN": twonn_dim,
        },
        ignore_index=True,
    )
    results = results.append(
        {
            # "PCA": pca_time,
            # "CPCA": cpca_time,
            "MLE": mle_time,
            "MLE (corrected)": mle_time_corrected,
            "TwoNN": twonn_time,
        },
        ignore_index=True,
    )

    results.to_csv(f"results_basenji{latent_dim}.csv")


else:
    if args.dataset == "MNIST":
        data = torchvision.datasets.MNIST(
            root="data/",
            train=True,
            transform=torchvision.transforms.ToTensor(),
            download=True,
        )
        print("Loaded MNIST dataset")
        data = data.data.reshape(len(data), -1).float()
        data = data / 255

    elif args.dataset == "CIFAR10":
        data = torchvision.datasets.CIFAR10(
            root="data/",
            train=True,
            transform=torchvision.transforms.ToTensor(),
            download=True,
        )
        print("Loaded CIFAR10 dataset")
        data = torch.from_numpy(data.data.reshape(len(data), -1))
        data = data.to(torch.float32)

        # Take 5000 random samples
        data = data[torch.randperm(data.shape[0])[:5000]]

    data = data.to(device)
    print("Data shape : ", data.shape)

    start = time.time()
    pca_dim = PCA(device=device).fit(data)
    print("PCA : ", pca_dim)
    pca_time = time.time() - start

    # start = time.time()
    # cpca_dim = CPCA(k=50, batch_size=batch_size, device=device).fit(data)
    # print("CPCA : ", cpca_dim)
    # cpca_time = time.time() - start

    start = time.time()
    mle_dim = MLE(k1=12, k2=20, batch_size=batch_size, device=device).fit(data)
    print("MLE : ", mle_dim)
    mle_time = time.time() - start

    start = time.time()
    mle_dim_corrected = MLE(
        k1=12, k2=20, mackay=True, batch_size=batch_size, device=device
    ).fit(data)
    print("MLE (corrected) : ", mle_dim_corrected)
    mle_time_corrected = time.time() - start

    start = time.time()
    twonn_dim = TwoNN(batch_size=batch_size, device=device).fit(data)
    print("TwoNN : ", twonn_dim)
    twonn_time = time.time() - start

    # results = pd.DataFrame(columns=["PCA", "CPCA", "MLE", "MLE (corrected)", "TwoNN"])
    results = pd.DataFrame(columns=["PCA", "MLE", "MLE (corrected)", "TwoNN"])

    results = results.append(
        {
            "PCA": pca_dim,
            # "CPCA": cpca_dim,
            "MLE": mle_dim,
            "MLE (corrected)": mle_dim_corrected,
            "TwoNN": twonn_dim,
        },
        ignore_index=True,
    )
    results = results.append(
        {
            "PCA": pca_time,
            # "CPCA": cpca_time,
            "MLE": mle_time,
            "MLE (corrected)": mle_time_corrected,
            "TwoNN": twonn_time,
        },
        ignore_index=True,
    )

    if args.dataset == "MNIST":
        results.to_csv("results_mnist.csv")
    elif args.dataset == "CIFAR10":
        results.to_csv("results_cifar10.csv")
