from dappy import read, preprocess
import numpy as np
import ssumo.data.quaternion as qtn
import torch
from ssumo.data.dataset import *

data_path = "/mnt/home/jwu10/working/ceph/data/immunostain/"
REORDER = [4, 3, 2, 1, 0, 5, 11, 10, 9, 8, 7, 6, 17, 16, 15, 14, 13, 12]
skeleton_config = read.config(
    "/mnt/home/jwu10/working/ssumo/configs/mouse_skeleton.yaml"
)
pose, ids = read.pose_h5(data_path + "pose_aligned.h5", dtype=np.float64)
data_path += "healthy/"
window = 51
dir_proc = "midfwd"
stride = 5

set_ids = (ids >= 37) if "healthy" in data_path else (ids < 37)
pose = pose[set_ids][:, REORDER, :]
## Smoothing
pose = preprocess.median_filter(pose, ids[set_ids], 5)
window_inds = get_window_indices(ids[set_ids], stride, window)

speed = get_speed_parts(
    pose=pose,
    parts=[
        [0, 1, 2, 3, 4, 5],  # spine and head
        [1, 6, 7, 8, 9, 10, 11],  # arms from front spine
        [5, 12, 13, 14, 15, 16, 17],  # left legs from back spine
    ],
)
speed = np.concatenate(
    [speed[:, :2], speed[:, 2:].mean(axis=-1, keepdims=True)], axis=-1
)
#[window_inds[:, 1:]].mean(axis=1)

# torch.save(torch.tensor(speed, dtype=torch.float32,), data_path + "avg_speed_3d.pt")
np.save(data_path + "speed_3d.npy", speed.astype(np.float32))
import pdb; pdb.set_trace()

if dir_proc == "fwd":
    yaw = get_frame_yaw(pose, 0, 1)[..., None]
else:
    windowed_yaw = get_frame_yaw(pose, 0, 1)[window_inds]
    yaw = windowed_yaw[:, window // 2][..., None]

root = pose[..., 0, :]
if dir_proc in ["midfwd", "x360"]:
    # Centering root
    root = pose[..., 0, :][window_inds]
    root_center = np.zeros(root.shape)
    root_center[..., [0, 1]] = root[:, window // 2, [0, 1]][:, None, :]

    root -= root_center
elif dir_proc == "fwd":
    root[..., [0, 1]] = 0

print("Applying inverse kinematics ...")
local_qtn = inv_kin(
    pose,
    skeleton_config["KINEMATIC_TREE"],
    np.array(skeleton_config["OFFSET"]),
    forward_indices=[1, 0],
)

if "fwd" in dir_proc:
    ## Center frame of a window is translated to center and rotated to x+

    if "mid" in dir_proc:
        fwd_qtn = np.zeros((len(window_inds), 4))
        fwd_qtn[:, [-1, 0]] = get_angle2D(yaw / 2)
        local_qtn = local_qtn[window_inds]
        fwd_qtn = np.repeat(fwd_qtn[:, None, :], window, axis=1)
    else:
        fwd_qtn = np.zeros((len(local_qtn), 4))
        fwd_qtn[:, [-1, 0]] = get_angle2D(yaw / 2)

    local_qtn[..., 0, :] = qtn.qmul_np(fwd_qtn, local_qtn[..., 0, :])

    root = qtn.qrot_np(fwd_qtn, root)

    # assert len(root) == len(window_inds)
    # assert len(local_qtn) == len(window_inds)

x6d = qtn.quaternion_to_cont6d_np(local_qtn)
print(x6d.shape)
np.save(data_path + "x6d_{}_s{}.npy".format(dir_proc, stride), x6d.astype(np.float32))
np.save(data_path + "root_{}_s{}.npy".format(dir_proc, stride), root.astype(np.float32))
# torch.save(
#     torch.tensor(x6d, dtype=torch.float32), data_path + "x6d_{}.pt".format(dir_proc)
# )
# torch.save(
#     torch.tensor(root, dtype=torch.float32), data_path + "root_{}.pt".format(dir_proc)
# )

offsets = get_segment_len(
    pose,
    skeleton_config["KINEMATIC_TREE"],
    np.array(skeleton_config["OFFSET"]),
)
# torch.save(torch.tensor(offsets, dtype=torch.float32), data_path + "offsets.pt")
np.save(data_path + "offsets.npy", offsets.astype(np.float32))

frame_dim_inds = tuple(range(len(root.shape) - 1))
print("Root Maxes: {}".format(root.max(axis=frame_dim_inds)))
print("Root Mins: {}".format(root.min(axis=frame_dim_inds)))


reshaped_x6d = x6d.reshape((-1,) + x6d.shape[-2:])
if dir_proc == "midfwd":
    offsets = offsets[window_inds].reshape(reshaped_x6d.shape[:2] + (-1,))
else:
    offsets = offsets

target_pose = fwd_kin_cont6d_torch(
    torch.tensor(reshaped_x6d, dtype=torch.float32),
    skeleton_config["KINEMATIC_TREE"],
    torch.tensor(offsets, dtype=torch.float32),
    root_pos=torch.zeros(reshaped_x6d.shape[0], 3),
    do_root_R=True,
    eps=1e-8,
).reshape(x6d.shape[:-1] + (3,))
# torch.save(target_pose, data_path + "target_pose_{}.pt".format(dir_proc))
np.save(
    data_path + "target_pose_{}_s{}.npy".format(dir_proc, stride),
    target_pose.detach().numpy().astype(np.float32),
)
