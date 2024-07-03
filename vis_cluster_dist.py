import ssumo
from dappy import read
import torch
from dappy import visualization as vis
import numpy as np
from pathlib import Path
from scipy.stats import circvar
import sys

from base_path import RESULTS_PATH

analysis_key = sys.argv[1]
vis_path = RESULTS_PATH + analysis_key + "/vis_latents/"
config = read.config(RESULTS_PATH + analysis_key + "/model_config.yaml")

dataset_label = "Train"
loader = ssumo.get.mouse_data(
    config=config,
    window=config["model"]["window"],
    train=dataset_label == "Train",
    # data_keys=["heading", "avg_speed"],
    data_keys=["view_axis"],
    shuffle=False,
)
dataset = loader.dataset

k_pred = np.load(vis_path + "z_gmm.npy")
keylist = ["view_axis"]  # change what the histograms are of
# keylist = ["heading", "ids"]

for key in keylist:
    # k_pred_null = np.load(vis_path + "z_{}_gmm.npy".format(key))
    if key == "heading":
        heading = dataset[:]["heading"].cpu().detach().numpy()
        feat = np.arctan2(heading[:, 0], heading[:, 1])
    if key == "view_axis":
        axes = config["data"]["project_axis"]
        feat = np.concatenate(
            [
                np.full(
                    int(len(dataset) / len(axes)), np.arctan2(axes[i][1], axes[i][2])
                )
                for i in range(len(axes))
            ]
        )
    else:
        feat = dataset[:][key].cpu().detach().numpy().squeeze()
    z_cvar, z_null_cvar = [], []
    for i in range(25):
        z_cvar += [circvar(feat[k_pred == i], high=np.pi, low=-np.pi)]
        # z_null_cvar += [circvar(feat[k_pred_null == i], high=np.pi, low=-np.pi)]

    print(np.mean(z_cvar))
    # print(np.mean(z_null_cvar))

    ssumo.plot.feature_ridge(
        feature=feat,
        labels=k_pred,
        xlabel=key,
        ylabel="Cluster",
        x_lim=(feat.min() - 0.1, feat.max() + 0.1),
        n_bins=200,
        binrange=(feat.min(), feat.max()),
        path="{}{}_".format(vis_path, key),
    )

    # ssumo.plot.feature_ridge(
    #     feature=feat,
    #     labels=k_pred_null,
    #     xlabel=key,
    #     ylabel="Cluster",
    #     x_lim=(feat.min() - 0.1, feat.max() + 0.1),
    #     binrange=(feat.min(), feat.max()),
    #     n_bins=200,
    #     path="{}{}_null_".format(vis_path, key),
    # )
