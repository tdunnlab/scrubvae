from neuroposelib import read
import numpy as np
import scrubvae.data.quaternion as qtn
from typing import List
import torch
from scrubvae.data.dataset import *
import h5py

def calculate_mouse_kinematics(
    data_path: dict,
    skeleton_config: dict,
    dataset: str,
    window: int = 51,
    train_val_test: str = "train",
    data_keys: List[str] = ["x6d", "root", "offsets"],
    remove_speed_outliers: float = 2.25,
    direction_process: str = "midfwd",
):
    print("Calculating dataset: {} {}".format(dataset, train_val_test))
    pose = np.load(
        "{}{}/{}/pose.npy".format(
            data_path, dataset, train_val_test
        )
    )
    ids = np.repeat(np.arange(4, dtype=int), pose.shape[0] // 4)

    # Filter out bad tracking using speed threshold
    if remove_speed_outliers is not None:
        outlier_frames = get_speed_outliers(
            pose, remove_speed_outliers
        )
        pose = np.delete(pose, outlier_frames, 0)
        ids = np.delete(ids, outlier_frames, 0)

    data = {"raw_pose": pose} if "raw_pose" in data_keys else {}
    # speed_keys = [key for key in data_keys if "speed" in key]
    # assert len(speed_keys) < 2
    # Calculate different speed representations
    if "avg_speed_3d" in data_keys:
        speed = get_speed_parts(
            pose=pose,
            parts=[
                [0, 1, 2, 3, 4, 5],  # spine and head
                [1, 6, 7, 8, 9, 10, 11],  # arms from front spine
                [5, 12, 13, 14, 15, 16, 17],  # left legs from back spine
            ],
        )
        data["avg_speed_3d"] = np.concatenate(
            [speed[:, :2], speed[:, 2:].mean(axis=-1, keepdims=True)], axis=-1
        )
        # elif "avg_speed" in speed_keys:
        #     speed = np.diff(pose, n=1, axis=-3) # Window axis
        #     speed = np.sqrt((speed**2).sum(axis=-1)).mean(axis=(-1, -2))

    # Get yaw of mid-spine -> fwd-spine segment for central frame in all windows
    yaw = get_frame_yaw(pose[:, window//2, ...], 0, 1)[..., None]

    # if "heading" in data_keys:
    data["heading"] = get_angle2D(yaw)

    # # Get yaw of center frame in a window
    # if direction_process in ["midfwd", "x360"]:
    #     yaw = yaw[window_inds][:, window // 2]

    if ("root" or "x6d") in data_keys:
        root = pose[..., 0, :] # (n_samples, window, 3)
        if direction_process in ["midfwd", "x360"]:
            # Centering root of middle frame
            # root = pose[..., 0, :][window_inds]
            root_center = np.zeros(root.shape) # (n_samples, window, 3)
            root_center[..., [0, 1]] = root[:, window // 2, [0, 1]][:, None, :] # (n_samples, 1, 2)

            root -= root_center
        # elif direction_process == "fwd":
        #     root[..., [0, 1]] = 0

    # Applying inverse kinematics to get local quaternions
    # That representation is the converted to a continuous 6D rotation
    if "x6d" in data_keys:
        print("Applying inverse kinematics ...")
        local_qtn = inv_kin(
            pose.reshape((-1,) + pose.shape[-2:]),
            skeleton_config["KINEMATIC_TREE"],
            np.array(skeleton_config["OFFSET"]),
            forward_indices=[1, 0],
        ).reshape(pose.shape[:-1] + (-1,))

        if direction_process == "midfwd":
            # if "mid" in direction_process:
                ## Center frame of a window is translated to center and rotated to x+
            fwd_qtn = np.zeros((len(yaw), 4))
            fwd_qtn[:, [-1, 0]] = get_angle2D(yaw / 2)
            # local_qtn = local_qtn[window_inds]
            fwd_qtn = np.repeat(fwd_qtn[:, None, :], window, axis=1)
            # else:
            #     # Otherwise just rotate all frames to x+
            #     fwd_qtn = np.zeros((len(local_qtn), 4))
            #     fwd_qtn[:, [-1, 0]] = get_angle2D(yaw / 2)

            local_qtn[..., 0, :] = qtn.qmul_np(fwd_qtn, local_qtn[..., 0, :])

            if "root" in data_keys:
                root = qtn.qrot_np(fwd_qtn, root)

            # assert len(root) == len(window_inds)
            # assert len(local_qtn) == len(window_inds)

        data["x6d"] = qtn.quaternion_to_cont6d_np(local_qtn)

    # Get offsets scaled by segment lengths
    if "offsets" in data_keys:
        data["offsets"] = get_segment_len(
            pose.reshape((-1,) + pose.shape[-2:]),
            skeleton_config["KINEMATIC_TREE"],
            np.array(skeleton_config["OFFSET"]),
        ).reshape(pose.shape)

    # Get root positions
    if "root" in data_keys:
        data["root"] = root
        frame_dim_inds = tuple(range(len(root.shape) - 1))
        print("Root Maxes: {}".format(root.max(axis=frame_dim_inds)))
        print("Root Mins: {}".format(root.min(axis=frame_dim_inds)))

    data = {k: torch.tensor(v, dtype=torch.float32) for k, v in data.items()}

    # Get animal IDs
    data["ids"] = torch.tensor(ids, dtype=torch.int16)

    # if "fluorescence" in data_keys:
    #     # Read in integrated fluorescence for PD dataset
    #     parent_path = str(Path(data_path).parents[0])
    #     import pandas as pd

    #     meta = pd.read_csv(parent_path + "/metadata.csv")
    #     meta_by_frame = meta.iloc[ids]
    #     fluorescence = meta_by_frame["Fluorescence"].to_numpy()
    #     data["fluorescence"] = torch.tensor(fluorescence, dtype=torch.float32)

    if "target_pose" in data_keys:
        # Target pose root does not move
        reshaped_x6d = data["x6d"].reshape((-1,) + data["x6d"].shape[-2:])
        # if direction_process == "midfwd":
        offsets = data["offsets"].reshape(
            reshaped_x6d.shape[:2] + (-1,)
        )
        # else:
        #     offsets = data["offsets"]

        data["target_pose"] = fwd_kin_cont6d_torch(
            reshaped_x6d,
            skeleton_config["KINEMATIC_TREE"],
            offsets,
            root_pos=torch.zeros(reshaped_x6d.shape[0], 3),
            do_root_R=True,
            eps=1e-8,
        ).reshape(data["x6d"].shape[:-1] + (3,))

    for k, v in data.items():
        full_path = "{}{}/{}/{}".format(
            data_path, dataset, train_val_test, k
        )
        if k in ["ids", "heading", "avg_speed_3d", "offsets"]:
            full_path += ".h5"
        else:
            full_path += "_{}.h5".format(direction_process)

        hf = h5py.File(full_path, "w")
        hf.create_dataset(k, data=v)
        hf.close()
        print(full_path)
        print(len(v))
        # np.save(full_path, v.numpy())

    return data


data_path = "/mnt/home/jwu10/working/ceph/data/wu_iclr25/"
data_keys = ["x6d", "root", "offsets", "target_pose", "avg_speed_3d", "heading", "ids"]
skeleton_config = read.config(data_path + "mouse_skeleton.yaml")

for label in ["train", "val", "test"]:
    for direction_process in ["midfwd", "x360"]:
        data = calculate_mouse_kinematics(
            data_path,
            skeleton_config,
            dataset="4_mice",
            window=51,
            train_val_test=label,
            data_keys=data_keys,
            remove_speed_outliers=2.25,
            direction_process=direction_process,
        )