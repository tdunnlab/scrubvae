from ssumo.data.dataset import fwd_kin_cont6d_torch

from torch.utils.data import DataLoader
from dappy import read
import torch
from dappy import vis
import ssumo
from base_path import RESULTS_PATH
import sys


"""
def visualize_reconstruction(model, loader, label, connectivity, config):
    n_keypts = loader.dataset.n_keypts
    kinematic_tree = loader.dataset.kinematic_tree
    model.eval()
    with torch.no_grad():
        # Let's see how reconstruction looks on train data
        data = next(iter(loader))
        data = next(iter(loader))
        data = {k: v.to("cuda") for k, v in data.items()}
        data_o = ssumo.train.predict_batch(
            model, data, disentangle_keys=config["disentangle"]["features"]
        )

        # pose_hat = fwd_kin_cont6d_torch(
        #     data_o["x6d"].reshape(-1, n_keypts, 6),
        #     kinematic_tree,
        #     data["offsets"].view(-1, n_keypts, 3),
        #     data_o["root"].reshape(-1, 3),
        #     do_root_R=True,
        # )

        pose_hat = fwd_kin_cont6d_torch(
            data["x6d"].reshape(-1, n_keypts, 6),
            kinematic_tree,
            data["offsets"].view(-1, n_keypts, 3),
            data["root"].reshape(-1, 3),
            do_root_R=True,
        )

        pose_array = torch.cat(
            [
                data["raw_pose"].reshape(-1, n_keypts, 3),  # true pose
                data["target_pose"].reshape(
                    -1, n_keypts, 3
                ),  # pose that the x6d must reconstruct
                pose_hat,  # true x6d reconstruction
            ],
            axis=0,
        )

        vis.pose.grid3D(
            pose_array.cpu().detach().numpy(),
            connectivity,
            frames=[
                0,
                config["data"]["batch_size"] * config["model"]["window"],
                2 * config["data"]["batch_size"] * config["model"]["window"],
            ],
            centered=False,
            subtitles=["Raw", "Target", "x6d fwd"],
            title=label + " Data",
            fps=45,
            figsize=(36, 12),
            N_FRAMES=config["data"]["batch_size"] * config["model"]["window"],
            VID_NAME=label + ".mp4",
            SAVE_ROOT=config["out_path"],
        )


config = [0, 0]
connectivity = [0, 0]
config[0] = read.config(RESULTS_PATH + "babelstride10" + "/model_config.yaml")
config[1] = read.config(RESULTS_PATH + "z128ch128d2_norot" + "/model_config.yaml")
config[0]["data"]["stride"] = 10
config[1]["data"]["stride"] = 10
config[0]["data"]["batch_size"] = 5
config[1]["data"]["batch_size"] = 5
connectivity[0] = read.connectivity_config(config[0]["data"]["skeleton_path"])
connectivity[1] = read.connectivity_config(config[1]["data"]["skeleton_path"])

dname = ["babel", "mouse"]
dataset = [0, 0]
model = [0, 0]
loader = [0, 0]

for n in range(2):
    dataset[n], loader[n], model[n] = ssumo.get.data_and_model(
        config[n],
        load_model=config[n]["out_path"],
        epoch=150,
        dataset_label="Train",
        data_keys=["x6d", "root", "offsets", "raw_pose", "target_pose"]
        + config[n]["disentangle"]["features"],
        shuffle=True,
        verbose=0,
        dataset_name=dname[n],
    )

    visualize_reconstruction(model[n], loader[n], "Train", connectivity[n], config[n])
import pdb

pdb.set_trace()

# fmt: off
from ssumo.train.losses import get_batch_loss
ne = 0
data = next(iter(loader[ne]))
torch.mean(torch.abs(torch.sum(data["offsets"][0][0], axis=1)))
data = {k: v.to("cuda") for k, v in data.items()}
data_o = ssumo.train.predict_batch(model[ne], data, disentangle_keys=config[ne]["disentangle"]["features"])
print(get_batch_loss(model[ne], data, data_o, config[ne]["loss"]))
# fmt: on

"""

import numpy as np
import matplotlib.pyplot as plt
import numpy as np
import random

skeleton_config = read.config(
    "/mnt/home/hkoneru/working/ssumo/configs/mouse_skeleton.yaml"
)

pose = read.pose_h5("/mnt/ceph/users/hkoneru/data/ensemble_healthy/pose_aligned.h5")
pose = np.array(pose[0])
REORDER = [4, 3, 2, 1, 0, 5, 11, 10, 9, 8, 7, 6, 17, 16, 15, 14, 13, 12]
pose = pose[:, REORDER, :]

"""
arrs = []
c = ["#D448A5", "#3F52C9", "#47A050"]
joints = [4, 11, 17]
joints = [4, 8, 14]
c = [
    skeleton_config["KEYPT_COLORS"][joints[0]],
    skeleton_config["KEYPT_COLORS"][joints[1]],
    skeleton_config["KEYPT_COLORS"][joints[2]],
]
plt.figure(figsize=(20, 10))
for k in range(9):
    y = pose[80000:120000, joints[k // 3], k % 3]
    x = np.arange(len(y))
    plt.subplot(9, 1, k + 1).axis("off")
    space = 1.2
    plt.ylim(min(y) * space, max(y) * space)
    plt.plot(x, y, color=c[k // 3], lw=2)

plt.savefig("motiontrace.png", dpi=100)
"""

index = [313826, 989197, 1002503]
# index = 0
az = [0, 35, 20]
n = len(index)
if index == 0:
    n = 10
    az = [0 for i in range(10)]

for j in range(n):
    figsize = (10, 10)
    fig = plt.figure(figsize=figsize)
    ax = plt.axes(projection="3d")
    ax.axis("off")
    ax.set_box_aspect([1, 1, 1])
    # import pdb

    # pdb.set_trace()
    ax.view_init(elev=5, azim=az[j])
    curr_frames = int(random.random() * 1295996)  # 1293503
    if index != 0:
        curr_frames = index[j]
    for index_from, index_to in skeleton_config["SEGMENTS"]:
        xs, ys, zs = [
            np.array(
                [
                    pose[curr_frames, index_from, j],
                    pose[curr_frames, index_to, j],
                ]
            )
            for j in range(3)
        ]
        # lw_color = np.sqrt(np.linspace(0, 0.75, 10))
        # linewidth = 3.5 - np.linspace(0, 3.1, 10)
        # for co, l in zip(lw_color, linewidth):
        plt.plot(
            xs,
            ys,
            zs,
            color="black",
            lw=5,
            alpha=0.8,  # - (i * 0.55 / full_pose_inds[-1])
        )

    for i in range(len(skeleton_config["KEYPT_COLORS"])):
        ax.scatter3D(
            pose[curr_frames, i, 0],
            pose[curr_frames, i, 1],
            pose[curr_frames, i, 2],
            marker="o",
            color=skeleton_config["KEYPT_COLORS"][i],
            s=500,
            alpha=1,  # 1 - (i * 0.75 / full_pose_inds[-1]),
            zorder=3.5,
        )

    plt.savefig(str(curr_frames) + "mouseskel.png", dpi=200)  # 300
