import numpy as np
from dappy import visualization as vis
import ssumo
import sys
from pathlib import Path
from dappy import read
import torch
from sklearn.linear_model import LinearRegression
from ssumo.data.dataset import *
from ssumo.plot import trace_grid
from tqdm import tqdm

RESULTS_PATH = "/mnt/ceph/users/hkoneru/results/vae/"
### Set/Load Parameters
key_id = 0
analysis_key = ["josh_heading", "josh_double_cond"][key_id]
epoch = [320, 280][key_id]
out_path = RESULTS_PATH + analysis_key
config = read.config(RESULTS_PATH + analysis_key + "/model_config.yaml")
config["model"]["load_model"] = config["out_path"]

vis_decode_path = RESULTS_PATH + analysis_key + "/vis_decode/"

Path(vis_decode_path).mkdir(parents=True, exist_ok=True)
connectivity = read.connectivity_config(config["data"]["skeleton_path"])
dataset_label = "Train"
### Load Datasets
loader, model = ssumo.get.data_and_model(
    config,
    load_model=config["out_path"],
    epoch=epoch,
    dataset_label=dataset_label,
    data_keys=["x6d", "root", "offsets", "avg_speed_3d", "heading"],
    shuffle=False,
    verbose=0,
)

latents = ssumo.get.latents(
    config=config,
    model=model,
    epoch=epoch,
    loader=loader,
    device="cuda",
    dataset_label=dataset_label,
)

# dis_w = LinearRegression().fit(latents, loader.dataset[:][disentangle_key]).coef_

n_shifts_s = 5
n_shifts_h = 5
dis_s = "avg_speed_3d"
dis_h = "heading"
# sample_idx = [1000]
sample_idx = [324406]
# sample_idx = [323237, 324406, 557490]
# sample_idx = torch.randint(low=0, high=len(loader.dataset), size=(30,))
shift_s = torch.linspace(-4.5, 1.5, n_shifts_s)[:, None].repeat(n_shifts_h, 1)
# shift_s = torch.linspace(-8, 8, n_shifts_s)[:, None].repeat(n_shifts_h, 1)
shift_h = torch.linspace(np.pi / 3, -np.pi / 3, n_shifts_h)[:, None].repeat_interleave(
    n_shifts_s
)


for sample_i in tqdm(sample_idx):
    data = loader.dataset[sample_i]
    data = {
        k: v.cuda()[None, ...].repeat(
            (n_shifts_s * n_shifts_h,)
            + tuple(np.ones(len(v.shape), dtype=int))
            # (n_shifts + 1,) + tuple(np.ones(len(v.shape), dtype=int))
        )
        for k, v in data.items()
    }
    z_traverse = (
        latents[sample_i : sample_i + 1, :].repeat(n_shifts_s * n_shifts_h, 1).cuda()
    )
    # z_traverse = latents[sample_i : sample_i + 1, :].repeat(n_shifts + 1, 1).cuda()
    data[dis_s] = data[dis_s] + shift_s.cuda()

    ang = torch.atan2(data[dis_h][:, 1], data[dis_h][:, 0]) + shift_h.cuda()
    data[dis_h] = torch.cat((torch.cos(ang)[:, None], torch.sin(ang)[:, None]), dim=1)
    # data[disentangle_key][1:, :] = data[disentangle_key][1:, :] + shift.cuda()

    data_o = model.decode(z_traverse, data)
    pose = (
        ssumo.data.dataset.fwd_kin_cont6d_torch(
            data_o["x6d"].reshape((-1,) + data_o["x6d"].shape[2:]),
            model.kinematic_tree,
            data["offsets"].reshape((-1,) + data["offsets"].shape[2:]),
            root_pos=data_o["root"].reshape((-1, 3)),
            do_root_R=True,
        )
        .detach()
        .cpu()
        .numpy()
    )
    shiftarr = torch.tensor([[[140.0, 140.0, 0.0]]]).repeat(
        n_shifts_h, n_shifts_s, model.window, loader.dataset.n_keypts, 1
    )
    shiftarr = shiftarr * torch.cat(
        (
            torch.cartesian_prod(torch.arange(n_shifts_h), torch.arange(n_shifts_s)),
            torch.zeros(n_shifts_h * n_shifts_s)[:, None],
        ),
        dim=1,
    ).reshape(n_shifts_h, n_shifts_s, 3)[:, :, None, None, :].expand(shiftarr.shape)
    pose = shiftarr.reshape((-1,) + shiftarr.shape[3:]) + pose

    trace_grid(
        pose,
        connectivity,
        vis_plane="xy",
        frames=np.arange(n_shifts_s * n_shifts_h) * model.window,
        n_full_pose=2,
        keypts_to_trace=[0, 4, 8, 11, 14, 17],
        centered=False,
        N_FRAMES=model.window,
        dpi=100,
        figure_size=(40, 40),
        FIG_NAME=dataset_label + "grid{}_mod.png".format(sample_i),
        SAVE_ROOT=vis_decode_path,
    )
