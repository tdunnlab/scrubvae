from neuroposelib.embed import Embed
import ssumo
from neuroposelib import read
from torch.utils.data import DataLoader
import numpy as np
import scipy.linalg as spl
from scripts.base_path import RESULTS_PATH
import matplotlib.pyplot as plt
import colorcet as cc
from ssumo.plot import scatter_cmap
import sys
from pathlib import Path

analysis_key = sys.argv[1]

if len(sys.argv) > 3:
    z_path = Path(RESULTS_PATH + analysis_key)
    folders = [str(f.parts[-1]) for f in z_path.iterdir() if f.is_dir()]
    job_id = sys.argv[3]
    analysis_key = "{}/{}/".format(analysis_key, folders[int(job_id)])

angles = 19
config = read.config(RESULTS_PATH + analysis_key + "/model_config.yaml")
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
    # data_keys=["x6d", "root", "heading"],
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
        recompute=False,
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
            np.abs(np.arctan2(axes[i][2], axes[i][1])),
        )
        for i in range(len(axes))
    ]
)

output_file = config["out_path"] + "tSNE_z_{}.npy".format(dataset_label)
if Path(output_file).exists():
    embed_vals = np.load(output_file)
else:
    embedder = Embed(
        embed_method="fitsne",
        perplexity=50,
        lr="auto",
    )
    embed_vals = embedder.embed(latents, save_self=True)
    np.save(output_file, embed_vals)


downsample = 10
rand_ind = np.random.permutation(np.arange(len(embed_vals)))
scatter_cmap(
    embed_vals[rand_ind, :][::downsample, :],
    feat[rand_ind][::downsample],
    "z_feat_{}".format(dataset_label),
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
