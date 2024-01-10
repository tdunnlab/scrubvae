from data.dataset import MouseDataset, fwd_kin_cont6d_torch, inv_normalize_root
import torch
from dappy import visualization as vis
import numpy as np
from pathlib import Path
import tqdm
import matplotlib.pyplot as plt
import utils
from dappy import read
import torch.optim as optim
from constants import PLANE
import plot

PARTS = ["Root", "Head + Spine", "L-Arm", "R-Arm", "L-Leg", "R-Leg"]

### Set/Load Parameters
path = "gr_parts1/partspd10_gre1_rc_w51_b1_midfwd_full"
base_path = "/mnt/home/jwu10/working/behavior_vae/"
config = read.config("/mnt/ceph/users/jwu10/results/vae/" + path + "/model_config.yaml")
config["load_model"] = config["out_path"]
config["load_epoch"] = 135
vis_decode_path = config["out_path"] + "/vis_decode/"
Path(vis_decode_path).mkdir(parents=True, exist_ok=True)
connectivity = read.connectivity_config(base_path + "/configs/mouse_skeleton.yaml")

dataset_label = "Train"
### Load Dataset
dataset = MouseDataset(
    data_path=config["data_path"],
    skeleton_path=base_path + "/configs/mouse_skeleton.yaml",
    train=True,
    window=config["window"],
    stride=1,
    direction_process=config["direction_process"],
    get_speed=config["speed_decoder"],
    get_root=config["arena_size"] is not None,
    get_raw_pose=False,
    arena_size=config["arena_size"],
)
arena_size = None if config["arena_size"] is None else dataset.arena_size.cuda()
kinematic_tree = dataset.kinematic_tree
n_keypts = dataset.n_keypts

vae, speed_decoder, device = utils.init_model(
    config, dataset.n_keypts, config["invariant"]
)
vae.eval()
speed_decoder.eval()
latents = utils.get_latents(vae, dataset, config, device, dataset_label)

if config["gradient_reversal"] == None:
    spd_w = speed_decoder.weight.cpu().detach()
else:
    spd_w = speed_decoder.decoder.weight.cpu().detach()


sample_idx = [2000, 200000, 400000, 600000]
norm_z_shift = spd_w / torch.linalg.norm(spd_w, axis=-1)[:, None]

minmax = 20
n_shifts = 15
graded_shift = torch.zeros((15, spd_w.shape[0]))
graded_shift[:,[2, 3]] = torch.linspace(-minmax, minmax, n_shifts)[:,None]
graded_shift[:,[4, 5]] = torch.linspace(minmax, -minmax, n_shifts)[:,None]

shift = (graded_shift @ norm_z_shift).cuda()

for sample_i in sample_idx:
    sample_latent = latents[sample_i : sample_i + 1].repeat(n_shifts, 1).cuda()

    sample_latent += shift

    if vae.in_channels == n_keypts * 6:
        x6d_o = vae.decoder(sample_latent)
        root_o = torch.zeros((n_shifts * config["window"], 3))
    elif vae.in_channels == n_keypts * 6 + 3:
        x_o = vae.decoder(sample_latent)
        x6d_o = x_o[:, :-3, :]
        root_o = x_o[:, -3:, :].moveaxis(-1, 1).reshape(-1, 3)
        root_o = inv_normalize_root(root_o, arena_size)

    offsets = dataset[:]["offsets"][sample_i : sample_i + 1].cuda()

    pose = (
        fwd_kin_cont6d_torch(
            x6d_o.moveaxis(-1, 1).reshape((-1, n_keypts, 6)),
            kinematic_tree,
            offsets.repeat(n_shifts, 1, 1, 1).reshape((-1, n_keypts, 3)),
            root_pos=root_o,
            do_root_R=True,
        )
        .cpu()
        .detach()
        .numpy()
    )

    subtitles = ["Arms: {:.2f}, Legs: {:.2f}".format(s, -s) for s in graded_shift[:, 2]]

    vis.pose.grid3D(
        pose,
        connectivity,
        frames=np.arange(n_shifts) * config["window"],
        centered=False,
        subtitles=subtitles,
        title=dataset_label + " Data - Speed Latent Shift",
        fps=15,
        N_FRAMES=config["window"],
        VID_NAME=dataset_label + "gridparts_{}.mp4".format(sample_i),
        SAVE_ROOT=vis_decode_path,
    )

    vis.pose.arena3D(
        pose,
        connectivity,
        frames=np.arange(n_shifts) * config["window"],
        centered=False,
        # subtitles=np.linspace(-minmax, minmax, n_shifts).astype(str),
        # title=dataset_label + " Data - Speed Latent Shift",
        fps=15,
        N_FRAMES=config["window"],
        VID_NAME=dataset_label + "arenaparts_{}.mp4".format(sample_i),
        SAVE_ROOT=vis_decode_path,
    )
