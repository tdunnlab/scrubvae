import ssumo
from neuroposelib import read
import numpy as np
import matplotlib.pyplot as plt
import sys
from scipy.spatial import procrustes

# from base_path import RESULTS_PATH

RESULTS_PATH = "/mnt/ceph/users/hkoneru/results/vae/"
stride = 20
angles = 25
analysis_key_3d = "180d_old/3d_vanilla"
epoch_3d = 80
redo_latent = True

analysis_key = sys.argv[1]
config = read.config(RESULTS_PATH + analysis_key + "/model_config.yaml")

config["data"]["project_axis"] = [
    np.round(np.array([0, -np.sin(i), -np.cos(i)]), 10)
    for i in np.linspace(0, np.pi, angles)
]
config["data"]["stride"] = stride

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
        recompute=redo_latent,
    )
    .cpu()
    .detach()
    .numpy()
)

axes = config["data"]["project_axis"]
feat = [
    np.round(np.abs(np.arctan2(axes[i][1], axes[i][2])), 3) for i in range(len(axes))
]
split_latents = latents.reshape((len(axes), -1, latents.shape[-1]))


config_3d = read.config(RESULTS_PATH + analysis_key_3d + "/model_config.yaml")

config_3d["data"]["stride"] = stride

### Load Datasets
loader_3d, model_3d = ssumo.get.data_and_model(
    config_3d,
    load_model=config_3d["out_path"],
    epoch=epoch_3d,
    dataset_label=dataset_label,
    data_keys=["x6d", "root", "view_axis"],
    shuffle=False,
    verbose=0,
)

latents_3d = (
    ssumo.get.latents(
        config_3d,
        model_3d,
        epoch_3d,
        loader_3d,
        device="cuda",
        dataset_label=dataset_label,
        recompute=redo_latent,
    )
    .cpu()
    .detach()
    .numpy()
)

split_latents_3d = np.repeat(latents_3d[None, ...], len(feat), 0)

procrustes_with_3D = [
    procrustes(split_latents[f], split_latents_3d[f])[2] for f in range(len(feat))
]

plt.plot(feat, procrustes_with_3D)
plt.scatter(feat, procrustes_with_3D)
ax = plt.gca()
# plt.xticks(np.arange(len(axes)))
# plt.yticks(np.arange(len(axes)))
# ax.set_xticklabels(feat)
plt.savefig("{}procrustes_with_3D.png".format(config["out_path"]), dpi=400)
plt.close()
