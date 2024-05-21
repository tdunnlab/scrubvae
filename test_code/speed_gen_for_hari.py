import numpy as np
from dappy import visualization as vis
import ssumo
import sys
from pathlib import Path
from dappy import read
import torch
from sklearn.linear_model import LinearRegression

RESULTS_PATH = "/mnt/ceph/users/jwu10/results/vae/"
### Set/Load Parameters
analysis_key = "mi_64_mf/bw50_500"
disentangle_key = "avg_speed_3d"
out_path = RESULTS_PATH + analysis_key
config = read.config(RESULTS_PATH + analysis_key + "/model_config.yaml")
config["model"]["load_model"] = config["out_path"]

#TODO: CHANGE THIS PATH
vis_decode_path = config["out_path"] + "/vis_decode/"

Path(vis_decode_path).mkdir(parents=True, exist_ok=True)
connectivity = read.connectivity_config(config["data"]["skeleton_path"])
dataset_label = "Train"
### Load Datasets
loader, model = ssumo.get.data_and_model(
    config,
    load_model=config["out_path"],
    epoch=270,
    dataset_label=dataset_label,
    data_keys=["x6d", "root", "offsets", disentangle_key],
    shuffle=False,
    verbose=0,
)

latents = ssumo.get.latents(
    config=config,
    model=model,
    epoch=270,
    loader=loader,
    device="cuda",
    dataset_label=dataset_label,
)

# dis_w = LinearRegression().fit(latents, loader.dataset[:][disentangle_key]).coef_

n_shifts = 10
sample_idx = [1000, 200000, 400000, 600000]
shift = torch.linspace(-10, 10, n_shifts)[:, None]

for sample_i in sample_idx:
    data = loader.dataset[sample_i]
    data = {
        k: v.cuda()[None, ...].repeat(
            (n_shifts + 1,) + tuple(np.ones(len(v.shape), dtype=int))
        )
        for k, v in data.items()
    }
    z_traverse = latents[sample_i : sample_i + 1, :].repeat(n_shifts + 1, 1).cuda()
    data[disentangle_key][1:, :] = data[disentangle_key][1:, :] + shift.cuda()

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

    ## TODO: Print the true average speed
    subtitles = [
        "{:2f}".format(val)
        for val in data["avg_speed_3d"].mean(-1).detach().cpu().numpy().squeeze()
    ]

    vis.pose.grid3D(
        pose,
        connectivity,
        frames=np.arange(n_shifts + 1) * model.window,
        centered=False,
        subtitles=subtitles,
        title=dataset_label + " Data - {} Traversal".format(disentangle_key),
        fps=20,
        N_FRAMES=model.window,
        VID_NAME=dataset_label + "grid{}_mod.mp4".format(sample_i),
        SAVE_ROOT=vis_decode_path,
    )

    #TODO: Adapt the `vis.pose.arena3D` code
    vis.pose.arena3D(
            pose,
            connectivity,
            frames=np.arange(n_shifts + 1) * model.window,
            centered=False,
            fps=15,
            N_FRAMES=model.window,
            VID_NAME=dataset_label + "arena{}_mod.mp4".format(sample_i),
            SAVE_ROOT=vis_decode_path,
        )
