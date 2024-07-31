import numpy as np
from ssumo.eval import cluster
from neuroposelib import read
import torch
import ssumo
from sklearn.mixture import GaussianMixture

CODE_PATH = "/mnt/home/jwu10/working/ssumo/"
RESULTS_PATH = "/mnt/ceph/users/jwu10/results/vae/"

vanilla_gmm = np.load(
    "/mnt/home/jwu10/working/ceph/results/vae/vanilla_64/2/vis_latents_train/z_300_gmm.npy"
)
downsample = 1

models = read.config(CODE_PATH + "configs/exp_finals.yaml")["avg_speed_3d"]
models = {m[0]: [m[1], m[2]] for m in models}
walking_list = [1, 4, 8, 38, 41, 44]

config = read.config(RESULTS_PATH + models["SC-VAE-MI"][0] + "/model_config.yaml")
loader = ssumo.get.mouse_data(
    data_config=config["data"],
    window=config["model"]["window"],
    train=True,
    data_keys=[
        "x6d",
        "root",
        "offsets",
        "avg_speed",
    ],
    normalize=[],
    shuffle=False,
)
feat = loader.dataset[:]["avg_speed"].detach().numpy()

walking_inds = np.in1d(vanilla_gmm,walking_list)
ssumo.plot.feature_ridge(
        feature=feat[walking_inds].squeeze(),
        labels=vanilla_gmm[walking_inds],
        xlabel="Average Speed",
        ylabel="Cluster",
        x_lim=(feat.min() - 0.1, feat.max() + 0.1),
        n_bins=200,
        binrange=(feat.min(), feat.max()),
        path="./results/walking_speed"
    )

for i, model_key in enumerate(models.keys()):
    path = "{}{}/".format(RESULTS_PATH, models[model_key][0])
    config = read.config(path + "/model_config.yaml")
    epoch = models[model_key][1]

    z = torch.tensor(
        np.load(path + "latents/Train_{}.npy".format(models[model_key][1]))[
            ::downsample
        ]
    )

    # k_pred = GaussianMixture(
    #     n_components=50,
    #     covariance_type="diag",
    #     max_iter=150,
    #     init_params = "k-means++",
    #     reg_covar=1e-5,
    #     verbose=1,
    # ).fit_predict(z)

    k_pred, gmm = ssumo.eval.cluster.gmm(
        latents=z,
        n_components=50,
        label="z50_{}".format(models[model_key][1]),
        path=path + "/vis_latents/",
        covariance_type="diag",
    )

    ssumo.plot.feature_ridge(
        feature=k_pred[walking_inds],
        labels=vanilla_gmm[walking_inds],
        xlabel="Average Speed",
        ylabel="Cluster",
        x_lim=(k_pred.min() - 0.1, k_pred.max() + 0.1),
        n_bins=50,
        binrange=(k_pred.min(), k_pred.max()),
        path="./results/walking_clusters_{}".format(model_key)
    )

