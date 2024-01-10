from data.dataset import (
    MouseDataset,
    MouseMidCentered,
    fwd_kin_cont6d_torch,
    inv_normalize_root,
)
from torch.utils.data import DataLoader
import torch
from dappy import visualization as vis
from pathlib import Path
import matplotlib.pyplot as plt
import numpy as np
import tqdm
from plot.constants import PALETTE_DICT
import utils
import pickle
from dappy import read

### Set/Load Parameters
path = "si_rc_w51_midfwd_full"
base_path = "/mnt/home/jwu10/working/behavior_vae/"
config = utils.read_config(base_path + "/results/" + path + "/model_config.yaml")
config["load_model"] = config["out_path"]
config["load_epoch"] = 190
vis_decode_path = config["out_path"] + "/vis_decode/"
Path(vis_decode_path).mkdir(parents=True, exist_ok=True)

connectivity = read.connectivity_config(base_path + "/configs/mouse_skeleton.yaml")
dataset_label = "Train"
### Load Dataset
dataset = MouseDataset(
    data_path=config["data_path"],
    skeleton_path=config["base_path"] + "/configs/mouse_skeleton.yaml",
    train=(dataset_label == "Train"),
    window=config["window"],
    stride=2,
    direction_process=config["direction_process"],
    arena_size=config["arena_size"],
    invariant=config["invariant"],
    speed=config["speed_decoder"] is not None,
)
loader = DataLoader(dataset=dataset, batch_size=4, shuffle=True)
kinematic_tree = dataset.kinematic_tree
n_keypts = dataset.n_keypts
arena_size = None if config["arena_size"] is None else dataset.arena_size.cuda()

vae, device = utils.init_model(config, n_keypts, config["invariant"])
vae.eval()
sample_idx = [1000, 100000, 200000, 300000]
data = {k: v.cuda() for k, v in dataset[sample_idx].items()}
x_i = torch.cat(
    (data["x6d"].view(data["x6d"].shape[:2] + (-1,)), data["root"]), axis=-1
)
if (config["invariant"] == "speed") or (config["speed_decoder"] == "avg"):
    data["speed"] = data["speed"].mean(dim=-1, keepdim=True)

x6d_o, mu, L = vae(x_i, invariant=data[config["invariant"]].cuda())
n_shifts = 10
if config["invariant"] == "direction":
    shifted_invariants = torch.linspace(-np.pi, np.pi, n_shifts, dtype=torch.float32)
    invariant = torch.zeros(n_shifts + 1, 2)
    invariant[1:, 0] = torch.cos(shifted_invariants)
    invariant[1:, 1] = torch.sin(shifted_invariants)
elif config["invariant"] == "speed":
    shifted_invariants = torch.linspace(0, 8, n_shifts, dtype=torch.float32)
    invariant = torch.zeros(n_shifts + 1, 1)
    invariant[1:, :] = shifted_invariants[:, None]


for i in range(len(sample_idx)):
    invariant[0, :] = data[config["invariant"]][i]

    x_o = vae.decoder(
        torch.cat((mu[i].repeat(n_shifts + 1, 1), invariant.cuda()), dim=-1)
    ).moveaxis(-1, 1)
    x6d_shifted = x_o[..., :-3].reshape((-1, n_keypts, 6))
    root_o = x_o[..., -3:].reshape(-1, 3)
    root_o = inv_normalize_root(root_o, arena_size)
    root = inv_normalize_root(data["root"][i].reshape(-1, 3), arena_size)

    pose = (
        fwd_kin_cont6d_torch(
            x6d_shifted,
            kinematic_tree,
            data["offsets"][i].repeat(n_shifts + 1, 1, 1, 1).reshape((-1, n_keypts, 3)),
            root_pos=root_o,  # torch.zeros((config["window"] * (n_shifts + 1), 3)).cuda(),
            do_root_R=True,
        )
        .cpu()
        .detach()
        .numpy()
    )
    pose_raw = (
        fwd_kin_cont6d_torch(
            data["x6d"][i].reshape((-1, n_keypts, 6)),
            kinematic_tree,
            data["offsets"][i].reshape((-1, n_keypts, 3)),
            root_pos=root,  # torch.zeros((config["window"], 3)).cuda(),
            do_root_R=True,
        )
        .cpu()
        .detach()
        .numpy()
    )

    vis.pose.grid3D(
        np.concatenate((pose_raw, pose), axis=0),
        connectivity,
        frames=np.arange(n_shifts + 2) * config["window"],
        centered=False,
        subtitles=np.concatenate(
            [["Raw", "Reconstructed"], shifted_invariants.detach().numpy().astype(str)]
        ),
        title="{} data - {} invariant shift".format(dataset_label,config["invariant"]),
        fps=25,
        N_FRAMES=config["window"],
        VID_NAME="{}_invrnt_grid{}.mp4".format(dataset_label, sample_idx[i]),
        SAVE_ROOT=vis_decode_path,
    )

    vis.pose.arena3D(
        pose[config["window"] :, ...],
        connectivity,
        frames=np.arange(n_shifts) * config["window"],
        centered=False,
        # subtitles=np.linspace(-minmax, minmax, n_shifts).astype(str),
        # title=dataset_label + " Data - Speed Latent Shift",
        fps=25,
        N_FRAMES=config["window"],
        VID_NAME="{}_invrnt_arena{}.mp4".format(dataset_label, sample_idx[i]),
        SAVE_ROOT=vis_decode_path,
    )
