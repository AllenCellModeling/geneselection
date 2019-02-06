import multiprocessing
from functools import partial

import numpy as np

from sklearn.preprocessing import scale
from sklearn.linear_model import enet_path

from .utils import get_selected_betas


def worker(
    boot_inds,
    X,
    y,
    X_noise=0.01,
    y_noise=0.5,
    alpha=0.9,
    lambda_path=np.geomspace(1.0, 0.01, num=100),
):

    X_boot = X[boot_inds, :]
    y_boot = y[boot_inds]

    X_boot = scale(
        scale(X_boot + np.random.normal(scale=X_noise * 1e-6, size=X_boot.shape))
        + np.random.normal(scale=X_noise, size=X_boot.shape)
    )
    y_boot = scale(
        scale(y_boot + np.random.normal(scale=y_noise * 1e-6, size=len(y_boot)))
        + np.random.normal(scale=y_noise, size=len(y_boot))
    )

    lambdas_enet, coefs_enet, _ = enet_path(
        X, y, l1_ratio=alpha, alphas=lambda_path, fit_intercept=False
    )

    return {"beta": get_selected_betas(coefs_enet), "lambda_path": lambdas_enet}


def parallel_runs(
    adata,
    n_processes=10,
    n_bootstraps=1000,
    X_noise=0.01,
    y_noise=0.5,
    alpha=0.9,
    lambda_path=np.geomspace(10, 0.01, num=10),
    day_offset=0.1,
):

    day_map = {
        "D0": 0,
        "D12": 1 - day_offset,
        "D14": 1 + day_offset,
        "D24": 2 - day_offset,
        "D26": 2 + day_offset,
        "D93": 3 - day_offset,
        "D96": 3 + day_offset,
    }

    boot_inds = [
        np.random.choice(len(adata.X), size=len(adata.X)) for _ in range(n_bootstraps)
    ]

    X = adata.X
    y = np.array([day_map[d] for d in adata.obs["day"]])

    worker_partial = partial(
        worker,
        X=X,
        y=y,
        X_noise=X_noise,
        y_noise=y_noise,
        alpha=alpha,
        lambda_path=lambda_path,
    )

    pool = multiprocessing.Pool(processes=n_processes)
    result_list = pool.map(worker_partial, boot_inds)
    pool.close()
    pool.join()

    return result_list
