import pickle
import numpy as np
from pathlib import Path

def get_gmm(
    latents: np.ndarray,
    n_components: int = 25,
    label: str = "cluster",
    path: str = "./results/",
    covariance_type: str = "full",
):
    model_exists = Path("{}{}_gmm.p".format(path, label)).exists()

    if model_exists:
        print("Found GMM model - Loading ...")
        model = pickle.load(open("{}{}_gmm.p".format(path, label), "rb"))
    else:
        print("\nNo GMM model found - Fitting sklearn GMM Model")
        from sklearn.mixture import GaussianMixture

        model = GaussianMixture(
            n_components=n_components,
            covariance_type=covariance_type,
            max_iter=150,
            verbose=1,
        ).fit(latents)

        pickle.dump(model, open("{}{}_gmm.p".format(path, label), "wb"))

    if Path("{}{}_pred.npy".format(path, label)).exists() and model_exists:
        print("Found existing clusterings - Loading ...")
        k_pred = np.load("{}{}_pred.npy".format(path, label))
    else:
        print("Calculating sklearn GMM clusters ...")
        k_pred = model.predict(latents)
        np.save("{}{}_pred.npy".format(path, label), k_pred)

    return k_pred, model