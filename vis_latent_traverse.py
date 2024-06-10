import numpy as np
from dappy import visualization as vis
import ssumo
import matplotlib.pyplot as plt
import sys
from pathlib import Path
from dappy import read
import torch
from sklearn.linear_model import LinearRegression
import random
from tqdm import tqdm
import pdb

disvar = 1  # 1 for heading, 0 for speed. also modify paths in get>data and eval>latents
RESULTS_PATH = "/mnt/ceph/users/hkoneru/results/vae/"
### Set/Load Parameters
analysis_key = ["josh_mi", "josh_heading"][disvar]
# analysis_key = "josh_mi"
disentangle_key = ["avg_speed_3d", "heading"][disvar]
# disentangle_key = "avg_speed_3d"
out_path = RESULTS_PATH + analysis_key
config = read.config(RESULTS_PATH + analysis_key + "/model_config.yaml")
config["model"]["load_model"] = config["out_path"]

vis_decode_path = config["out_path"] + "vis_decode/"

Path(vis_decode_path).mkdir(parents=True, exist_ok=True)
connectivity = read.connectivity_config(config["data"]["skeleton_path"])
dataset_label = "Train"
### Load Datasets
dataset, loader, model = ssumo.get.data_and_model(
    config,
    load_model=config["out_path"],
    # epoch=270,
    epoch=[270, 320][disvar],  # 270 for speed, 320 for heading
    dataset_label=dataset_label,
    data_keys=[
        ["x6d", "root", "offsets", disentangle_key],
        ["x6d", "root", "offsets", disentangle_key, "avg_speed_3d"],
    ][disvar],
    shuffle=False,
    verbose=0,
)


latents = ssumo.get.latents(
    config=config,
    model=model,
    # epoch=270,
    epoch=[270, 320][disvar],
    dataset=dataset,
    # loader=loader,
    device="cuda",
    dataset_label=dataset_label,
)


# dis_w = LinearRegression().fit(latents, loader.dataset[:][disentangle_key]).coef_

n_shifts = 3

# sample_idx = [int(random.random() * len(dataset)) for i in range(20)]
# sample_idx = [625290, 318272, 296269, 617192, 682264]  # for speed accordion
# sample_idx = [318272, 296269, 625290]  # for speed
sample_idx = [[318272, 296269, 625290], [625290, 296269, 682264]][1]  # for heading

linspace = torch.linspace(-torch.pi / 2, torch.pi / 2, n_shifts)[:, None].cuda()
circ = torch.cat([torch.sin(linspace), torch.cos(linspace)], dim=-1)
shift = torch.linspace(-3, 3, n_shifts)[:, None]
posefull = np.zeros((0, 18, 3))
subtitles = []
for sample_i in tqdm(sample_idx):
    data = loader.dataset[sample_i]
    data = {
        k: v.cuda()[None, ...].repeat(
            (n_shifts + 1,) + tuple(np.ones(len(v.shape), dtype=int))
        )
        for k, v in data.items()
    }
    z_traverse = latents[sample_i : sample_i + 1, :].repeat(n_shifts + 1, 1).cuda()
    if disvar:
        data[disentangle_key][1:, :] = circ.cuda()
    else:
        data[disentangle_key][1:, :] = data[disentangle_key][1:, :] + shift.cuda()
        # # data[disentangle_key][1:, 0] = data[disentangle_key][1:, 0] + shift[:, 0].cuda()

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
    posefull = np.concatenate((posefull, pose[51:]))

    ## TODO: Print the true average speed
    # subtitles += [
    #     "{:2f}".format(val)
    #     # for val in data["avg_speed_3d"][..., 0].detach().cpu().numpy().squeeze()
    #     for val in data["avg_speed_3d"].mean(-1).detach().cpu().numpy().squeeze()
    # ][1:]

posefull = posefull.reshape(9, 51, 18, 3)[[0, 3, 6, 1, 4, 7, 2, 5, 8]].reshape(
    -1, 18, 3
)

subtitles = [
    [
        "Slowed",
        "",
        "",
        "Original",
        "",
        "",
        "Sped",
        "",
        "",
    ],
    [
        "Left",
        "",
        "",
        "Original",
        "",
        "",
        "Right",
        "",
        "",
    ],
][disvar]

vis.pose.grid3D(
    posefull,
    connectivity,
    frames=np.arange(n_shifts**2) * model.window,
    centered=False,
    subtitles=subtitles,
    # title=dataset_label + " Data - {} Traversal".format(disentangle_key),
    title=[
        "Conditional Speed Motion Generation",
        "Conditional Heading Motion Generation",
    ][disvar],
    fps=20,
    dpi=400,
    N_FRAMES=model.window,
    VID_NAME=dataset_label + ["gridfull_mod.mp4", "gridfull_heading.mp4"][disvar],
    SAVE_ROOT=vis_decode_path,
)

"""

for sample_i in tqdm(sample_idx):
    data = loader.dataset[sample_i]
    data = {
        k: v.cuda()[None, ...].repeat(
            (n_shifts + 1,) + tuple(np.ones(len(v.shape), dtype=int))
        )
        for k, v in data.items()
    }
    z_traverse = latents[sample_i : sample_i + 1, :].repeat(n_shifts + 1, 1).cuda()
    data[disentangle_key][1:, :] = data[disentangle_key][1:, :] + shift.cuda()
    # data[disentangle_key][1:, 0] = data[disentangle_key][1:, 0] + shift[:, 0].cuda()

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
        for val in data["avg_speed_3d"][..., 0].detach().cpu().numpy().squeeze()
        # for val in data["avg_speed_3d"].mean(-1).detach().cpu().numpy().squeeze()
    ]

    height = 120
    figsize = (10, 25)
    fig = plt.figure(figsize=figsize)
    ax = plt.axes(projection="3d")
    ax.axis("off")
    from mpl_toolkits.mplot3d.axes3d import Axes3D

    # ax.get_proj = lambda: np.dot(Axes3D.get_proj(ax), np.diag([2, 2, 2, 1]))
    ax.set_box_aspect([1, 1, 2.5])
    ax.view_init(elev=5, azim=0)
    plotframes = sum([[0 + 51 * i, 50 + 51 * i] for i in range(1, n_shifts + 1)], [])
    for fr in range(len(plotframes)):
        curr_frames = plotframes[fr]
        for index_from, index_to in connectivity.links:
            xs, ys, zs = [
                np.array(
                    [
                        pose[curr_frames, index_from, j],
                        pose[curr_frames, index_to, j],
                    ]
                )
                for j in range(3)
            ]
            plt.plot(
                xs,
                ys,
                zs + (fr // 2) * height,
                color=(
                    (fr // 2) / (n_shifts),
                    0,
                    1 - (fr // 2) / (n_shifts),
                ),
                lw=3,
                alpha=0.5,
            )
        if fr % 2 == 0:
            for k in range(50):
                for jo in [4, 5, 8, 11, 14, 17]:
                    xs, ys, zs = [
                        np.array(
                            [
                                pose[plotframes[fr] + k, jo, j],
                                pose[plotframes[fr] + k + 1, jo, j],
                            ]
                        )
                        for j in range(3)
                    ]
                    plt.plot(
                        xs,
                        ys,
                        zs + (fr // 2) * height,
                        color=(
                            (fr // 2) / (n_shifts),
                            0,
                            1 - (fr // 2) / (n_shifts),
                        ),
                        lw=1,
                        alpha=0.5,
                    )

        for i in range(len(connectivity.keypt_colors)):
            ax.scatter3D(
                pose[curr_frames, i, 0],
                pose[curr_frames, i, 1],
                pose[curr_frames, i, 2] + (fr // 2) * height,
                marker="o",
                color=connectivity.keypt_colors[i],
                s=3,
                alpha=1,  # 1 - (i * 0.75 / full_pose_inds[-1]),
                zorder=3.5,
            )

    plt.savefig(str(sample_i) + "mouseskel.png", dpi=400)  # 300
    """

# # TODO: Adapt the `vis.pose.arena3D` code
# vis.pose.arena3D(
#     pose,
#     connectivity,
#     frames=np.arange(n_shifts + 1) * model.window,
#     centered=False,
#     fps=15,
#     N_FRAMES=model.window,
#     VID_NAME=dataset_label + "arena{}_mod.mp4".format(sample_i),
#     SAVE_ROOT=vis_decode_path,
# )
