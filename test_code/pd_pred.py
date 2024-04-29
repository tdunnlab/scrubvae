import ssumo
from torch.utils.data import DataLoader
from dappy import read
import torch
from dappy import visualization as vis
import numpy as np
from pathlib import Path
import sys
from sklearn.model_selection import cross_val_score
from sklearn.linear_model import LogisticRegression

RESULTS_PATH = "/mnt/ceph/users/jwu10/results/vae/"
analysis_key = sys.argv[1]
config = read.config(RESULTS_PATH + analysis_key + "/model_config.yaml")
config["data"]["stride"] = 10
connectivity = read.connectivity_config(config["data"]["skeleton_path"])
dataset_label = "Train"
### Load Datasets
loader, model = ssumo.get.data_and_model(
    config,
    load_model=config["out_path"],
    epoch=sys.argv[2],
    dataset_label=dataset_label,
    data_keys=["x6d", "root", "avg_speed_3d", "fluorescence"],
    shuffle=False,
    verbose=0,
    normalize = ["avg_speed_3d"],
)
speed = loader.dataset[:]["avg_speed_3d"].cpu().detach().numpy()

latents = ssumo.get.latents(
    config=config,
    model=model,
    epoch=sys.argv[2],
    loader=loader,
    device="cuda",
    dataset_label=dataset_label,
)

pd_label = np.array(loader.dataset[:]["fluorescence"] < 0.9, dtype=int).ravel()
clf = LogisticRegression(solver="sag", max_iter=200)
scores = cross_val_score(clf, latents, pd_label)
print("Latents PD Prediction")
print(scores)

scores = cross_val_score(clf, speed, pd_label)
print("Speed PD Prediction")
print(scores)

scores = cross_val_score(clf, np.concatenate([latents, speed], axis=-1), pd_label)
print("Latent and Speed PD Prediction")
print(scores)
