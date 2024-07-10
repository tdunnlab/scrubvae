from dappy.embed import Embed
import ssumo
from dappy import read
from torch.utils.data import DataLoader
import numpy as np
import scipy.linalg as spl
import matplotlib.pyplot as plt
from cmocean.cm import phase
import colorcet as cc
from ssumo.plot import scatter_cmap
import sys
from sklearn.decomposition import PCA

RESULTS_PATH = "/mnt/ceph/users/hkoneru/results/vae/"

analysis_key = sys.argv[1]
config = read.config(RESULTS_PATH + analysis_key + "/model_config.yaml")

dataset_label = "Train"
### Load Datasets
loader, model = ssumo.get.data_and_model(
    config,
    load_model=config["out_path"],
    epoch=sys.argv[2],
    dataset_label=dataset_label,
    # data_keys=["x6d", "root", "heading"],
    data_keys=["x6d", "root", "view_axis"],
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

# heading = dataset[:]["heading"].cpu().detach().numpy()
# feat = np.arctan2(heading[:, 0], heading[:, 1])

axes = config["data"]["project_axis"]
feat = np.concatenate(
    [
        np.full(
            int(len(loader.dataset) / len(axes)),
            np.abs(np.arctan2(axes[i][1], axes[i][2])),
        )
        for i in range(len(axes))
    ]
)

downsample = 10

pca = PCA(n_components=2)
embed_vals = pca.fit_transform(latents[::downsample, :].astype(np.float64))


rand_ind = np.random.permutation(np.arange(len(embed_vals)))
scatter_cmap(
    embed_vals[rand_ind, :],
    feat[::downsample][rand_ind],
    "pca_z_feat_{}".format(dataset_label),
    path=config["out_path"],
    cmap="viridis",
)


# k_pred = np.load(config["out_path"] + "vis_latents/z_gmm.npy")
# scatter_cmap(embed_vals[::downsample, :], k_pred[::downsample], "gmm", path=config["out_path"], cmap=plt.get_cmap("gist_rainbow"))

# z_null = ssumo.eval.project_to_null(
#     z, model.disentangle["heading"].decoder.weight.detach().cpu().numpy()
# )[0]

# # embed_vals = embedder.embed(z_null, save_self=True)
# # np.save(config["out_path"] + "tSNE_znull.npy", embed_vals)

# # embed_vals = np.load(config["out_path"] + "tSNE_znull.npy")

# scatter_cmap(
#     embed_vals[::downsample, :], yaw[::downsample], "znull_yaw", path=config["out_path"]
# )
