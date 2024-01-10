from dataset import MouseDataset,fwd_kin_cont6d_torch,inv_normalize_root
from dappy import visualization as vis
from dappy import read
import numpy as np


connectivity = read.connectivity_config("./configs/mouse_skeleton.yaml")
path = "rc_w51_midfwd_full"
base_path = "/mnt/ceph/users/jwu10/results/vae/"
out_path = base_path + path + "/vis_latents_sub/"
config = read.config(base_path + path + "/model_config.yaml")

dataset = MouseDataset(
    data_path=config["data_path"],
    skeleton_path="/mnt/home/jwu10/working/behavior_vae/configs/mouse_skeleton.yaml",
    train=True,
    window=config["window"],
    stride=1000,
    direction_process=config["direction_process"],
    get_speed=config["speed_decoder"] is not None,
    arena_size=config["arena_size"],
    invariant=config["invariant"],
    get_raw_pose=True,)

idx = 1
data = dataset[idx]

pose_converted = fwd_kin_cont6d_torch(
    data["x6d"],
    dataset.kinematic_tree,
    data["offsets"],
    root_pos=inv_normalize_root(data["root"], dataset.arena_size),
    do_root_R=True,
).numpy()

vis.pose.arena3D(
    pose_converted,
    connectivity,
    frames=[0],
    centered=False,
    fps=45,
    N_FRAMES=config["window"],
    dpi=50,
    VID_NAME="factor_test.mp4",
    SAVE_ROOT="./",
)

import pdb; pdb.set_trace()

print("Calculating forward kinematics ...")
n_keypts = dataset.n_keypts
pose_converted = fwd_kin_cont6d_torch(
    dataset.data["local_6d"].reshape(-1, n_keypts, 6),
    dataset.kinematic_tree,
    dataset.data["offsets"][dataset.window_inds].reshape(-1, n_keypts, 3),
    root_pos=inv_normalize_root(dataset.data["root"].reshape(-1, 3), dataset.arena_size),
    do_root_R=True,
).numpy()

pose_converted = pose_converted.reshape(-1, config["window"], n_keypts, 3)

print("Checking assertions ...")
assert (
    np.linalg.norm( pose_converted[:, config["window"] // 2, 1, 1] - pose_converted[:, config["window"] // 2, 0, 1] )
    < 0.1
)

assert 0 < np.mean(pose_converted[..., 14, 2]) < np.mean(pose_converted[..., 0, 2])

assert 0 < np.mean(pose_converted[..., 17, 2]) < np.mean(pose_converted[..., 0, 2])

assert 0 < np.mean(pose_converted[..., 8, 2])

assert 0 < np.mean(pose_converted[..., 11, 2])

print("Finished assertions")


recon_roots = inv_normalize_root(dataset.data["root"].reshape(-1, 3), dataset.arena_size)
print("Root Mins")
print(recon_roots.min(axis=0))

print("Root Maxes")
print(recon_roots.max(axis=0))