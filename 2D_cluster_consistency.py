import ssumo
from neuroposelib import read
import torch
from neuroposelib import visualization as vis
import numpy as np
from pathlib import Path
from scipy.stats import circvar
import sys
import matplotlib.pyplot as plt

from scripts.base_path import RESULTS_PATH

angles = 19
stride = 10
maxlabels = 19
redo_latents = False
key = "view_axis"
analysis_key = sys.argv[1]
vis_path = RESULTS_PATH + analysis_key + "/vis_latents/"
Path(vis_path).mkdir(parents=True, exist_ok=True)
config = read.config(RESULTS_PATH + analysis_key + "/model_config.yaml")

dataset_label = "Train"
axes = [
    np.round(np.array([0, -np.sin(i), -np.cos(i)]), 10)
    for i in np.linspace(0, np.pi, angles)
]
config["data"]["project_axis"] = axes
config["data"]["stride"] = stride

loader, model = ssumo.get.data_and_model(
    config,
    load_model=config["out_path"],
    epoch=sys.argv[2],
    dataset_label=dataset_label,
    data_keys=["x6d", "root", "view_axis", "offsets", "segment_lens"],
    shuffle=False,
    verbose=0,
)
dataset = loader.dataset

latents = ssumo.get.latents(
    config,
    model,
    sys.argv[2],
    loader,
    device="cuda",
    dataset_label=dataset_label,
    recompute=redo_latents,
)

Path(config["out_path"] + "vis_latents/").mkdir(parents=True, exist_ok=True)

k_pred, gmm = ssumo.eval.cluster.gmm(
    latents=latents,
    n_components=25,
    label="z",
    path=vis_path,
    covariance_type="diag",
)

split_latents = latents.reshape((len(axes), -1, latents.shape[-1]))
split_k_pred = k_pred.reshape((len(axes), -1))

cluster_consistency = [
    [
        sum(split_k_pred[v1] == split_k_pred[v2]) / split_k_pred.shape[-1]
        for v1 in range(angles)
    ]
    for v2 in range(angles)
]

feat = [
    np.round(np.abs(np.arctan2(axes[i][1], axes[i][2])), 2) for i in range(len(axes))
]

stepsize = int(np.ceil(angles / maxlabels))

plt.imshow(cluster_consistency)
ax = plt.gca()
plt.xticks(np.arange(angles))
plt.xticks(rotation=90)
plt.yticks(np.arange(angles))
ax.set_xticklabels([f if f in feat[::stepsize] + [feat[-1]] else "" for f in feat])
ax.set_yticklabels([f if f in feat[::stepsize] + [feat[-1]] else "" for f in feat])
plt.colorbar()
plt.savefig("{}2D_cluster_consistency.png".format(config["out_path"]), dpi=400)
plt.close()
