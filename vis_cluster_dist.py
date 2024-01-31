import ssumo
from torch.utils.data import DataLoader
from dappy import read
import torch
from dappy import visualization as vis
import numpy as np
from pathlib import Path

from base_path import RESULTS_PATH

path = "/heading/balanced/"
vis_path = RESULTS_PATH + path + "/vis_latents_300/"
config = read.config(RESULTS_PATH + path + "/model_config.yaml")
config["model"]["load_model"] = config["out_path"]
config["model"]["start_epoch"] = 300

dataset_label = "Train"
dataset = ssumo.data.get_mouse(
    data_config=config["data"],
    window=config["model"]["window"],
    train=dataset_label == "Train",
    data_keys=["x6d", "root", "offsets", "raw_pose"]
    + config["disentangle"]["features"],
)
loader = DataLoader(
    dataset=dataset, batch_size=config["train"]["batch_size"], shuffle=False
)
k_pred = np.load(vis_path + "z_gmm.npy")

for key in ["heading", "avg_speed"]:
    k_pred_null = np.load(vis_path + "z_{}_gmm.npy".format(key))
    if key == "heading":
        heading = dataset[:]["heading"].cpu().detach().numpy()
        feat = np.arctan2(heading[:, 1], heading[:, 0])
    else:
        feat = dataset[:][key].cpu().detach().numpy().squeeze()

    ssumo.plot.feature_ridge(
        feature=feat,
        labels=k_pred,
        xlabel=key,
        ylabel="Cluster",
        x_lim = (feat.min() - 0.1, feat.max() + 0.1),
        n_bins=200,
        binrange = (feat.min(), feat.max()),
        path="{}{}_".format(vis_path, key),
    )

    ssumo.plot.feature_ridge(
        feature=feat,
        labels=k_pred_null,
        xlabel=key,
        ylabel="Cluster",
        x_lim = (feat.min() - 0.1, feat.max() + 0.1),
        binrange = (feat.min(), feat.max()),
        n_bins=200,
        path="{}{}_null_".format(vis_path, key),
    )
