from dappy.embed import Embed
import ssumo
from dappy import read
from torch.utils.data import DataLoader
import numpy as np
import scipy.linalg as spl
from base_path import RESULTS_PATH
import matplotlib.pyplot as plt
from cmocean.cm import phase
import colorcet as cc
from ssumo.plot import scatter_cmap

path = "/mcmi_32/vanilla/"
config = read.config(RESULTS_PATH + path + "/model_config.yaml")
config["model"]["load_model"] = config["out_path"]
config["model"]["start_epoch"] = 600

dataset_label = "Train"
dataset, loader = ssumo.data.get_mouse(
    data_config=config["data"],
    window=config["model"]["window"],
    train=dataset_label=="Train",
    data_keys=["x6d", "root", "heading"],  # + config["disentangle"]["features"],
    normalize=["heading"],
)

heading = dataset[:]["heading"].cpu().detach().numpy()
yaw = np.arctan2(heading[:, 1], heading[:, 0])

vae, device = ssumo.model.get(
    model_config=config["model"],
    disentangle_config=config["disentangle"],
    n_keypts=dataset.n_keypts,
    direction_process=config["data"]["direction_process"],
    arena_size=dataset.arena_size,
    kinematic_tree=dataset.kinematic_tree,
    bound=config["data"]["normalize"] is not None,
    verbose=-1,
)
z = (
    ssumo.eval.get.latents(vae, dataset, config, device, dataset_label)
    .cpu()
    .detach()
    .numpy()
)
embedder = Embed(
    embed_method="fitsne",
    perplexity=50,
    lr="auto",
)
embed_vals = embedder.embed(z, save_self=True)
np.save(config["out_path"] + "tSNE_z.npy", embed_vals)

# embed_vals = np.load(config["out_path"] + "tSNE_z.npy")

downsample = 10
scatter_cmap(
    embed_vals[::downsample, :], yaw[::downsample], "z_yaw", path=config["out_path"]
)

# z_null = ssumo.eval.project_to_null(
#     z, vae.disentangle["heading"].decoder.weight.detach().cpu().numpy()
# )[0]

# # embed_vals = embedder.embed(z_null, save_self=True)
# # np.save(config["out_path"] + "tSNE_znull.npy", embed_vals)

# # embed_vals = np.load(config["out_path"] + "tSNE_znull.npy")

# scatter_cmap(
#     embed_vals[::downsample, :], yaw[::downsample], "znull_yaw", path=config["out_path"]
# )
