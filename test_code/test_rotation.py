import numpy as np
import torch
import ssumo
from ssumo.data import quaternion as qtn
from neuroposelib import read
from neuroposelib import visualization as vis
from ssumo.data.dataset import get_frame_yaw

RESULTS_PATH = "/mnt/ceph/users/jwu10/results/vae/"
analysis_key = "mals_64/cvae_64"
config = read.config(RESULTS_PATH + analysis_key + "/model_config.yaml")
connectivity = read.connectivity_config(
    "/mnt/home/jwu10/working/ssumo/configs/mouse_skeleton.yaml"
)

loader = ssumo.get.mouse_data(
    data_config=config["data"],
    window=config["model"]["window"],
    train=True,
    data_keys=[
        "x6d",
        "root",
        "offsets",
        "avg_speed_3d",
        "heading",
        "raw_pose",
    ],
    normalize=[],
    shuffle=False,
)
heading_true = loader.dataset[:]["heading"]
raw_pose = loader.dataset[:]["raw_pose"].numpy()
raw_pose -= raw_pose[:, config["model"]["window"]//2, 0, :][:, None, None, :]
yaw_true = np.arctan2(heading_true[:, 0].numpy(), heading_true[:, 1].numpy())
heading1D_rand = (torch.rand(len(loader.dataset)) * 2 - 1)[:, None] * np.pi
rot_angle = (-heading1D_rand.ravel().numpy() + yaw_true)[::5]

rot_mat = np.array(
    [
        [np.cos(rot_angle), -np.sin(rot_angle), np.zeros(len(rot_angle))],
        [np.sin(rot_angle), np.cos(rot_angle), np.zeros(len(rot_angle))],
        [np.zeros(len(rot_angle)), np.zeros(len(rot_angle)), np.ones(len(rot_angle))],
    ]
).repeat(loader.dataset.n_keypts*config["model"]["window"], axis=2)

pose_rot = np.einsum("jki,ik->ij", rot_mat, raw_pose[::5].reshape(-1, 3)).reshape(
        raw_pose[::5].shape
    )

idx = [1000, 4000, 50000]
pose_cat = np.concatenate((raw_pose[::5][idx], pose_rot[idx]),axis=0)
print(np.degrees(get_frame_yaw(pose_cat[:,config["model"]["window"]//2,...], 0, 1)))
print(np.degrees(heading1D_rand[::5][idx].numpy()))
print(np.degrees(yaw_true[::5][idx]))
vis.pose.arena3D(
    pose_cat.reshape(-1, loader.dataset.n_keypts, 3),
    connectivity,
    frames=[0, len(idx) * config["model"]["window"]],
    centered=False,
    N_FRAMES=len(idx)*config["model"]["window"],
    fps=30,
    dpi=100,
    VID_NAME="test.mp4",
    SAVE_ROOT="./",
)

import pdb; pdb.set_trace()
