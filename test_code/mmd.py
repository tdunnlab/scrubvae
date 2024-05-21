import numpy as np
from scipy.spatial.distance import cdist, pdist
import ssumo
from dappy import read
import sys
import matplotlib.pyplot as plt

def mmd_estimate(X, Y, h=None):
    """
    Given samples from two distributions in a common
    common feature space, this function computes an
    estimate of the maximal mean discrepancy (MMD)
    distance with a squared exponential kernel.

    Reference
    ---------
    Gretton et al. (2012). A Kernel Two-Sample Test.
    Journal of Machine Learning Research 13: 723-773.

    Parameters
    ----------
    X : ndarray (num_x_samples x num_features)
        First set of observed samples, assumed to be
        drawn from some unknown distribution P.

    Y : ndarray (num_y_samples x num_features)
        Second set of observed samples, assumed to be
        drawn from some unknown distribution Q.

    h : float
        Bandwidth parameter

    Returns
    -------
    dist : float
        An unbiased estimator of the MMD.
    """

    # Compute pairwise distances
    xd = pdist(X, metric="euclidean")
    yd = pdist(Y, metric="euclidean")
    xyd = cdist(X, Y, metric="euclidean").ravel()

    if h is None:
        h = np.median(np.concatenate((xd, yd, xyd))) ** 2
    # Compute unbiased MMD distance estimate.
    kxx = np.mean(np.exp(-(xd**2) / h))
    kyy = np.mean(np.exp(-(yd**2) / h))
    kxy = np.mean(np.exp(-(xyd**2) / h))
    return kxx + kyy - 2 * kxy


RESULTS_PATH = "/mnt/ceph/users/jwu10/results/vae/"
palette = ssumo.plot.constants.PALETTE_2
# task_id = sys.argv[1] if len(sys.argv)>1 else ""
analysis_key = sys.argv[1]
# analysis_keys = ["vanilla_64", "cvae_64", "mi_64_new", "mals_64"]#, "mi_64_new", "mi_64_fixed"]
epoch = 150
# if task_id.isdigit():
#     analysis_keys = [analysis_keys[int(task_id)]]

config = read.config(RESULTS_PATH + analysis_key + "/model_config.yaml")

loader = ssumo.get.mouse_data(
    data_config = config["data"],
    window = config["model"]["window"],
    train=True,
    data_keys=["avg_speed_3d", "fluorescence", "ids"],
    shuffle=False,
    normalize=["avg_speed_3d"],
)
speed = loader.dataset[:]["avg_speed_3d"].cpu().detach().numpy()
ids = loader.dataset[:]["ids"].cpu().detach().numpy()
fluorescence = loader.dataset[:]["fluorescence"].cpu().detach().numpy()
latents = (
    ssumo.get.latents(
        config=config,
        epoch=epoch,
        loader=loader,
        device="cuda",
        dataset_label="Train",
    )
    .cpu()
    .detach()
    .numpy()
)

mmd_animal = {"Latents": [], "Speed": [], "Both": [], "Fluorescence": []}
for i in range(ids.max() + 1):
    pd_i = ((fluorescence < 0.9) & (ids == i)).ravel()
    healthy_i = ((fluorescence >= 0.9) & (
        ids == i
    )).ravel()

    pd_speed_i = speed[pd_i, ...]
    healthy_speed_i = speed[healthy_i, ...]
    pd_z_i = latents[pd_i, ...]
    healthy_z_i = latents[healthy_i, ...]

    mmd_animal["Latents"] += [mmd_estimate(pd_z_i, healthy_z_i)]
    mmd_animal["Speed"] += [mmd_estimate(pd_speed_i, healthy_speed_i)]
    mmd_animal["Both"] += [
        mmd_estimate(
            np.concatenate([pd_z_i, pd_speed_i], axis=-1),
            np.concatenate([healthy_z_i, healthy_speed_i], axis=-1),
        )
    ]
    mmd_animal["Fluorescence"] += [fluorescence[ids==i][0]]
    print(mmd_animal)


f, ax = plt.subplots(1, 3, figsize=(10,5))
f.suptitle("MMD For {}".format(analysis_key))
for i, key in enumerate(["Latents","Speed","Both"]):
    ax[i].scatter(mmd_animal["Fluorescence"], mmd_animal[key])
    ax[i].set_title(key.title())
    ax[i].set_ylabel("MMD")
    ax[i].set_xlabel("Fluorescence")

plt.savefig(config["out_path"] + "./mmd.png")
plt.close()
