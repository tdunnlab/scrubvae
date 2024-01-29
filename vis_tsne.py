from dappy.embed import Embed
import ssumo
from dappy import visualization as vis
from dappy import read
from torch.utils.data import DataLoader
import numpy as np
import scipy.linalg as spl

path = "balanced"
base_path = "/mnt/ceph/users/jwu10/results/vae/heading/"
config = read.config(base_path + path + "/model_config.yaml")
config["model"]["load_model"] = config["out_path"]
config["model"]["start_epoch"] = 300

dataset = ssumo.data.get_mouse(
    data_config=config["data"],
    window=config["model"]["window"],
    train="Test",
    data_keys=["x6d", "root", "heading"],  # + config["disentangle"]["features"],
)
loader = DataLoader(
    dataset=dataset, batch_size=config["train"]["batch_size"], shuffle=False
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
    verbose=-1,
)
z = (
    ssumo.evaluate.get.latents(vae, dataset, config, device, "Test")
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

vis.plot.scatter_by_cat(embed_vals, yaw, label="z_yaw", filepath=config["out_path"])

dis_w = vae.disentangle["heading"].decoder.weight.detach().cpu().numpy()
U_orth = spl.null_space(dis_w)
z_sub = z @ U_orth

embed_vals = embedder.embed(z, save_self=True)

vis.plot.scatter_by_cat(embed_vals, yaw, label="zsub_yaw", filepath=config["out_path"])
