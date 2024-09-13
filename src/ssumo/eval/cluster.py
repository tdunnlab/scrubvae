import pickle
import numpy as np
from pathlib import Path
import functools
from typing import Optional

def _check_model_exists(func):
    @functools.wraps(func)
    def wrapper(
        latents: np.ndarray,
        label: str = "cluster",
        path: Optional[str] = None,
        **kwargs,
    ):
        if path is None:
            model_exists = None
        else:
            model_path = "{}{}_{}.p".format(path, label, func.__name__)
            preds_path = "{}{}_{}.npy".format(path,label, func.__name__,)
            model_exists = Path(model_path).exists()

        if model_exists:
            print("Found {} model - Loading ...".format(func.__name__))
            model = pickle.load(open(model_path, "rb"))
        else:
            model = func(
                latents=latents,
                **kwargs,
            )
            if path is not None:
                print("Saving GMM model")
                pickle.dump(model, open(model_path, "wb"))

        if model_exists:
            if Path(preds_path).exists():
                print("Found existing {} clusterings - Loading ...".format(func.__name__))
                k_pred = np.load(preds_path)
        else:
            print("Calculating sklearn {} clusters ...".format(func.__name__))
            k_pred = model.predict(latents)
            if path is not None:
                print("Saving GMM cluster predictions")
                np.save(preds_path, k_pred)

        return k_pred, model

    return wrapper


@_check_model_exists
def gmm(
    latents: np.ndarray,
    n_components: int = 25,
    covariance_type: str = "full",
):
    from sklearn.mixture import GaussianMixture

    model = GaussianMixture(
        n_components=n_components,
        covariance_type=covariance_type,
        max_iter=150,
        init_params = "k-means++",
        reg_covar=1e-5,
        verbose=1,
    ).fit(latents)

    return model

def dbscan(
    latents: np.ndarray,
    eps: float = 0.1,
    min_samples= 500,
    label: str = "cluster",
    path: str = "./results/",
):
    preds_path = "{}{}_sc_pred.npy".format(path,label)
    # from sklearn.decomposition import PCA

    # latents = PCA(n_components=50).fit_transform(latents)
    from sklearn.cluster import HDBSCAN
    print("Calculating sklearn dbscan clusters ...")
    k_pred = HDBSCAN(min_cluster_size=min_samples).fit_predict(latents)
    print(len(np.unique(k_pred)))
    # k_pred = model.predict(latents)
    np.save(preds_path, k_pred)

    return k_pred