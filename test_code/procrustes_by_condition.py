import ssumo
from neuroposelib import read
import numpy as np
import matplotlib.pyplot as plt
import sys
from scipy.spatial import procrustes

# from base_path import RESULTS_PATH

RESULTS_PATH = "/mnt/ceph/users/hkoneru/results/vae/"

analysis_key = sys.argv[1]
config = read.config(RESULTS_PATH + analysis_key + "/model_config.yaml")
angles = 19
config["data"]["stride"] = 10
axes = [
    np.round(np.array([0, -np.sin(i), -np.cos(i)]), 10)
    for i in np.linspace(0, np.pi, angles)
]
config["data"]["project_axis"] = axes

dataset_label = "Train"
### Load Datasets
loader, model = ssumo.get.data_and_model(
    config,
    load_model=config["out_path"],
    epoch=sys.argv[2],
    dataset_label=dataset_label,
    data_keys=["x6d", "root", "view_axis", "offsets"],
    shuffle=False,
    verbose=0,
)

latents = (
    ssumo.get.latents(
        config,
        model,
        sys.argv[2],
        loader,
        device="cuda",
        dataset_label=dataset_label,
    )
    .cpu()
    .detach()
    .numpy()
)

axes = config["data"]["project_axis"]
feat = [
    np.round(-np.abs(np.arctan2(axes[i][1], axes[i][2])), 3) for i in range(len(axes))
]
split_latents = latents.reshape((len(axes), -1, latents.shape[-1]))
procrustes_matrix = np.zeros((len(axes), len(axes)))

for i in range(len(axes)):
    for j in range(len(axes)):
        if j < i:
            procrustes_matrix[i, j] = procrustes_matrix[j, i]
        elif j > i:
            procrustes_matrix[i, j] = procrustes(split_latents[i], split_latents[j])[2]

plt.imshow(procrustes_matrix, vmax=1)
ax = plt.gca()
plt.xticks(np.arange(len(axes)))
plt.yticks(np.arange(len(axes)))
ax.set_xticklabels(feat)
ax.set_yticklabels(feat)
plt.xticks(rotation=90)
plt.colorbar()
plt.savefig("{}procrustes_by_view.png".format(config["out_path"]), dpi=400)
plt.close()
