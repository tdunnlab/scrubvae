from dappy import read, preprocess
import numpy as np
import ssumo
from dappy import visualization as vis

config = read.config(
    "/mnt/ceph/users/jwu10/results/vae/heading/balanced/model_config.yaml"
)
connectivity = read.connectivity_config("../../../configs/mouse_skeleton.yaml")
data_config = config["data"]

REORDER = [4, 3, 2, 1, 0, 5, 11, 10, 9, 8, 7, 6, 17, 16, 15, 14, 13, 12]
skeleton_config = read.config(data_config["skeleton_path"])
pose, ids = read.pose_h5(data_config["data_path"], dtype=np.float64)
pose = pose[:, REORDER, :]

pose_rot = preprocess.rotate_spine(
    preprocess.center_spine(pose, keypt_idx=5), keypt_idx=[5, 4]
)

vis.pose.arena3D(
    pose_rot[(pose_rot[:, 1, 0] - pose_rot[:, 0, 0]) < 0, :, :],
    connectivity,
    frames=0,
    centered=False,
    fps=10,
    N_FRAMES=((pose_rot[:, 1, 0] - pose_rot[:, 0, 0]) < 0).sum(),
    dpi=100,
    VID_NAME="test2.mp4",
    SAVE_ROOT="./",
)
