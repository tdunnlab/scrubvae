from neuroposelib import read, preprocess
import numpy as np
import ssumo.data.quaternion as qtn
from typing import List
import torch
from ssumo.data.dataset import *
from torch.utils.data import DataLoader
from pathlib import Path

TRAIN_IDS = {
    # "immunostain": np.arange(74),
    # "immunostain": np.array([1,6,35,36,38,43,72,73]),
    "immunostain": np.array([18, 19, 21, 36, 55, 56, 58, 73]),
    "ensemble_healthy": np.arange(3),
    "neurips_mouse": np.arange(9),
}


def get_babel(
    data_config: dict,
    window: int = 51,
    train: bool = True,
    data_keys: List[str] = ["x6d", "root", "offsets"],
    shuffle: bool = False,
):
    """Read in BABEL dataset (cite)

    Args:
        data_config (dict): Dict of parameter options for loading in data.
        window (int, optional): # of frames per sample to be given by DataLoader class. Defaults to 51.
        train (bool, optional): Read in train or test set. Defaults to True.
        data_keys (List[str], optional): Data fields to be given by DataLoader class. Defaults to ["x6d", "root", "offsets"].
        shuffle (bool, optional): Whether DataLoader shuffles. Defaults to False.
    """
    ## TODO: Before starting this, set up your new mouse skeleton config
    # You can reference `ssumo/configs/mouse_skeleton.yaml``
    # Crucial elements are to identify your kinematic tree and offsets
    # Offsets will just be unit vectors in principle directions based off of kinematic tree
    # May also need to reorder keypoints
    skeleton_config = read.config(data_config["skeleton_path"])

    # TODO: Load in pose (Frames x keypoints x 3) and ids (1 per video) similar to neuroposelib
    pose, ids = None, None  # Load or index train or test

    # Save raw pose in dataset if specified
    data = {"raw_pose": pose} if "raw_pose" in data_keys else {}

    # Get windowed indices (n_frames x window)
    window_inds = get_window_indices(ids, data_config["stride"], window)

    # Get root xyz position and center on (0,0,z)
    root = pose[..., 0, :][window_inds]
    root_center = np.zeros(root.shape)
    root_center[..., [0, 1]] = root[:, window // 2, [0, 1]][:, None, :]

    # Get local 6D rotation representation -
    # Zhou, Yi, et al. "On the continuity of rotation representations in
    # neural networks." Proceedings of the IEEE/CVF Conference on
    # Computer Vision and Pattern Recognition. 2019.
    if "x6d" in data_keys:
        print("Applying inverse kinematics ...")
        # Getting local quaternions
        local_qtn = inv_kin(
            pose,
            skeleton_config["KINEMATIC_TREE"],
            np.array(skeleton_config["OFFSET"]),
            forward_indices=[
                None,
                None,
            ],  # TODO: Change these to defined forward vector
        )
        # Converting quaternions to 6d rotation representations
        data["x6d"] = qtn.quaternion_to_cont6d_np(local_qtn)

    # Scale offsets by segment lengths
    if "offsets" in data_keys:
        data["offsets"] = get_segment_len(
            pose,
            skeleton_config["KINEMATIC_TREE"],
            np.array(skeleton_config["OFFSET"]),
        )

    # Move everything to tensors
    data = {k: torch.tensor(v, dtype=torch.float32) for k, v in data.items()}

    # Initialize Dataset and DataLoaders
    dataset = MouseDataset(
        data,
        window_inds,
        data_config["arena_size"],
        skeleton_config["KINEMATIC_TREE"],
        pose.shape[-2],
        label="Train" if train else "Test",
    )
    loader = DataLoader(
        dataset=dataset,
        batch_size=data_config["batch_size"],
        shuffle=shuffle,
        num_workers=5,
        pin_memory=True,
    )

    return dataset, loader


# Use quaternions and fwd/inv kinematic functions to do the following:
# 1. Apply inv kin -> local 6D rotation -> fwd kinematics to reconstruct the original pose sequences
# 2. Once obtaining the decomposed local 6D rotations, visualize purely (0,0,0)-centered pose sequences
#    using only the fwd kinematic function, i.e. do not apply a translation anywhere.
# 3. Factor out yaw angle from the quaternions such that when you apply fwd kinematics, all poses will
#    face in the x+ direction, and be (0,0,0)-centered as in (2.)
# 3. Again using only the fwd kin function, plot pose sequences such that all segments are of length 1.
# 4. Hardest one: rotate each window of pose sequences (length 51) such that the MIDDLE POSE is centered
#    on (0,0,Z) and rotated to face in the x+ direction. The other poses in the sequence should be rotated,
#    and translated accordingly.

# Models to train - a couple of the best VAE model architectures that you've found.
# Use pose representation which in which you only center the middle pose on (0,0,Z).
# Translate other poses according.


def mouse_pd_data(
    data_config: dict,
    window: int = 51,
    data_keys: List[str] = ["x6d", "root", "offsets"],
):
    REORDER = [4, 3, 2, 1, 0, 5, 11, 10, 9, 8, 7, 6, 17, 16, 15, 14, 13, 12]
    if (
        (data_config["remove_speed_outliers"] is not None)
        or ("ids" in data_keys)
        or ("raw_pose" in data_keys)
    ):
        pose, ids = read.pose_h5(data_config["data_path"], dtype=np.float64)
        pose = pose[..., REORDER, :]

    parent_path = str(Path(data_config["data_path"]).parents[0])
    subfolders = ["/6ohda/", "/healthy/"]

    window_inds = get_window_indices(ids, data_config["stride"], window)
    if data_config["remove_speed_outliers"] is not None:
        outlier_frames = get_speed_outliers(
            pose, window_inds, data_config["remove_speed_outliers"]
        )
        kept_frames = np.arange(len(window_inds), dtype=np.int)
        kept_frames = np.delete(kept_frames, outlier_frames, 0)
        window_inds = window_inds[kept_frames]
        print("Window Inds: {}".format(window_inds.shape))

    saved_tensors = ["avg_speed_3d", "offsets", "root", "target_pose", "x6d"]
    data = {k: [] for k in data_keys if k in saved_tensors}
    for key in data.keys():
        print("Loading in {} data".format(key))
        for subfolder in subfolders:
            if key == "avg_speed_3d":
                npy_path = "{}{}speed_3d".format(parent_path, subfolder)
                data[key] += [np.load(npy_path + ".npy")]
            elif key != "offsets":
                npy_path = "{}{}{}_{}_s{}".format(
                    parent_path,
                    subfolder,
                    key,
                    data_config["direction_process"],
                    data_config["stride"],
                )
                data[key] += [np.load(npy_path + ".npy")]
            else:
                npy_path = "{}{}{}".format(parent_path, subfolder, key)
                data[key] += [np.load(npy_path + ".npy")]

        data[key] = np.concatenate(data[key], axis=0)
        data[key] = torch.tensor(data[key], dtype=torch.float32)

        if key == "avg_speed_3d":
            data[key] = data[key][window_inds[:, 1:]].mean(axis=1)

        if (data[key].shape[1] == window) and (
            data_config["remove_speed_outliers"] is not None
        ):
            data[key] = data[key][kept_frames]

    if "raw_pose" in data_keys:
        data["raw_pose"] = torch.tensor(pose, dtype=torch.float32)

    if "fluorescence" in data_keys:
        import pandas as pd

        meta = pd.read_csv(parent_path + "/metadata.csv")
        meta_by_frame = meta.iloc[ids]
        fluorescence = meta_by_frame["Fluorescence"].to_numpy()[window_inds[:, 0:1]]
        data["fluorescence"] = torch.tensor(fluorescence, dtype=torch.float32)

    if "ids" in data_keys:
        ids[ids >= 37] -= 37
        data["ids"] = torch.tensor(ids[window_inds[:, 0:1]], dtype=torch.int16)

    for k, v in data.items():
        print("{}: {}".format(k, v.shape))

    return data, window_inds


def mouse_data(
    data_config: dict,
    window: int = 51,
    train: bool = True,
    data_keys: List[str] = ["x6d", "root", "offsets"],
    shuffle: bool = False,
    normalize: List[str] = [],
):
    """_summary_

    Args:
        data_config (dict): _description_
        window (int, optional): _description_. Defaults to 51.
        train_ids (List, optional): _description_. Defaults to [0, 1, 2].
        train (bool, optional): _description_. Defaults to True.
        data_keys (List[str], optional): _description_. Defaults to ["x6d", "root", "offsets"].
        shuffle (bool, optional): _description_. Defaults to False.

    Returns:
        _type_: _description_
    """

    skeleton_config = read.config(data_config["skeleton_path"])

    if "immunostain" in data_config["data_path"]:
        if (data_config["stride"] == 5) or (data_config["stride"] == 10):
            data, window_inds = mouse_pd_data(data_config, window, data_keys)
        else:
            data, window_inds = calculate_mouse_kinematics(
                data_config,
                skeleton_config,
                window,
                train,
                data_keys,
            )
    else:
        data, window_inds = calculate_mouse_kinematics(
            data_config,
            skeleton_config,
            window,
            train,
            data_keys,
        )

    for key in normalize:
        if (key not in ["heading", "ids", "fluorescence"]) and (key in data_keys):
            if data_config["normalize"] == "bounded":
                print(
                    "Rescaling decoding variable, {}, to be between -1 and 1".format(
                        key
                    )
                )
                
                key_min = data[key].min(dim=0)[0] * 0.9
                data[key] -= key_min
                key_max = data[key].max(dim=0)[0] * 1.1
                data[key] = 2 * data[key] / key_max - 1
                assert data[key].max() < 1
                assert data[key].min() > -1
            elif data_config["normalize"] == "z_score":
                print(
                    "Mean centering and unit standard deviation-scaling {}".format(key)
                )
                data[key] -= data[key].mean(axis=0)
                data[key] /= data[key].std(axis=0)

    discrete_classes = {}
    if "ids" in data.keys():
        if "immunostain" in data_config["data_path"]:
            if not ((data_config["stride"] == 5) or (data_config["stride"] == 10)):
                data["ids"][data["ids"] >= 37] -= 37
                unique_ids = torch.unique(data["ids"])
                discrete_classes["ids"] = torch.arange(len(unique_ids)).long()
                for id in unique_ids:
                    data["ids"][data["ids"] == id] = discrete_classes["ids"][
                        id == unique_ids
                    ]
            else:
                discrete_classes["ids"] = torch.unique(data["ids"], sorted=True)
        else:
            discrete_classes["ids"] = torch.unique(data["ids"], sorted=True)

    dataset = MouseDataset(
        data,
        window_inds,
        data_config["arena_size"],
        skeleton_config["KINEMATIC_TREE"],
        len(skeleton_config["LABELS"]),
        label="Train" if train else "Test",
        discrete_classes=discrete_classes,
    )

    loader = DataLoader(
        dataset=dataset,
        batch_size=data_config["batch_size"],
        shuffle=shuffle,
        num_workers=5,
        pin_memory=True,
    )

    return loader


def calculate_mouse_kinematics(
    data_config: dict,
    skeleton_config: dict,
    window: int = 51,
    train: bool = True,
    data_keys: List[str] = ["x6d", "root", "offsets"],
):
    REORDER = [4, 3, 2, 1, 0, 5, 11, 10, 9, 8, 7, 6, 17, 16, 15, 14, 13, 12]
    pose, ids = read.pose_h5(data_config["data_path"], dtype=np.float64)

    for k in TRAIN_IDS.keys():
        if k in data_config["data_path"]:
            train_ids = TRAIN_IDS[k]

    set_ids = np.in1d(ids, train_ids) if train else ~np.in1d(ids, train_ids)
    pose = pose[set_ids][:, REORDER, :]
    ids = ids[set_ids]

    window_inds = get_window_indices(ids, data_config["stride"], window)

    ## Smoothing
    if data_config["filter_pose"]:
        pose = preprocess.median_filter(pose, ids, 5)

    data = {"raw_pose": pose} if "raw_pose" in data_keys else {}

    if data_config["remove_speed_outliers"] is not None:
        outlier_frames = get_speed_outliers(
            pose, window_inds, data_config["remove_speed_outliers"]
        )
        window_inds = np.delete(window_inds, outlier_frames, 0)

    yaw = get_frame_yaw(pose, 0, 1)[..., None]

    if "heading_change" in data_keys:
        heading2D = get_angle2D(yaw)
        data["heading_change"] = np.sqrt(
            (np.diff(heading2D[window_inds], n=1, axis=1) ** 2).sum(axis=1)
        )

    if "heading" in data_keys:
        data["heading"] = get_angle2D(yaw[window_inds][:, window // 2])

    if data_config["direction_process"] in ["midfwd", "x360"]:
        yaw = yaw[window_inds][:, window // 2]  # [..., None]

    if ("root" or "x6d") in data_keys:
        root = pose[..., 0, :]
        if data_config["direction_process"] in ["midfwd", "x360"]:
            # Centering root
            root = pose[..., 0, :][window_inds]
            root_center = np.zeros(root.shape)
            root_center[..., [0, 1]] = root[:, window // 2, [0, 1]][:, None, :]

            root -= root_center
        elif data_config["direction_process"] == "fwd":
            root[..., [0, 1]] = 0

    if "x6d" in data_keys:
        print("Applying inverse kinematics ...")
        local_qtn = inv_kin(
            pose,
            skeleton_config["KINEMATIC_TREE"],
            np.array(skeleton_config["OFFSET"]),
            forward_indices=[1, 0],
        )

        if "fwd" in data_config["direction_process"]:
            if "mid" in data_config["direction_process"]:
                print(
                    "Preprocessing poses such that the central pose\npasses through the origin in the x+ direction"
                )
                ## Center frame of a window is translated to center and rotated to x+
                fwd_qtn = np.zeros((len(window_inds), 4))
                fwd_qtn[:, [-1, 0]] = get_angle2D(yaw / 2)
                local_qtn = local_qtn[window_inds]
                fwd_qtn = np.repeat(fwd_qtn[:, None, :], window, axis=1)

            else:
                fwd_qtn = np.zeros((len(local_qtn), 4))
                fwd_qtn[:, [-1, 0]] = get_angle2D(yaw / 2)

            local_qtn[..., 0, :] = qtn.qmul_np(fwd_qtn, local_qtn[..., 0, :])

            if "root" in data_keys:
                root = qtn.qrot_np(fwd_qtn, root)

            if "mid" in data_config["direction_process"]:
                assert len(root) == len(window_inds)
                assert len(local_qtn) == len(window_inds)
            else:
                assert len(local_qtn) == len(pose)

        data["x6d"] = qtn.quaternion_to_cont6d_np(local_qtn)

    offsets = get_segment_len(
        pose,
        skeleton_config["KINEMATIC_TREE"],
        np.array(skeleton_config["OFFSET"]),
    )
    if "offsets" in data_keys:
        data["offsets"] = offsets

    if "root" in data_keys:
        data["root"] = root
        frame_dim_inds = tuple(range(len(root.shape) - 1))
        print("Root Maxes: {}".format(root.max(axis=frame_dim_inds)))
        print("Root Mins: {}".format(root.min(axis=frame_dim_inds)))

    data = {k: torch.tensor(v, dtype=torch.float32) for k, v in data.items()}

    if "ids" in data_keys:
        data["ids"] = torch.tensor(ids[window_inds[:, 0:1]], dtype=torch.int16)

    if "fluorescence" in data_keys:
        parent_path = str(Path(data_config["data_path"]).parents[0])
        import pandas as pd

        meta = pd.read_csv(parent_path + "/metadata.csv")
        meta_by_frame = meta.iloc[ids]
        fluorescence = meta_by_frame["Fluorescence"].to_numpy()[window_inds[:, 0:1]]
        data["fluorescence"] = torch.tensor(fluorescence, dtype=torch.float32)

    speed_key = [key for key in data_keys if "speed" in key]
    assert len(speed_key) < 2
    if ("target_pose" in data_keys) or (len(speed_key) > 0):
        reshaped_x6d = data["x6d"].reshape((-1,) + data["x6d"].shape[-2:])
        if data_config["direction_process"] == "midfwd":
            offsets = torch.tensor(
                offsets[window_inds].reshape(reshaped_x6d.shape[:2] + (-1,)),
                dtype=torch.float32,
            )
        else:
            offsets = torch.tensor(offsets, dtype=torch.float32)

        target_pose = fwd_kin_cont6d_torch(
            reshaped_x6d,
            skeleton_config["KINEMATIC_TREE"],
            offsets,
            root_pos=torch.zeros(reshaped_x6d.shape[0], 3),
            do_root_R=True,
            eps=1e-8,
        ).reshape(data["x6d"].shape[:-1] + (3,))

        if data_config["direction_process"] == "midfwd":
            target_pose[:, 25, 1, 1] = 0
        else:
            target_pose[:, 1, 1] = 0

    if "target_pose" in data_keys:
        data["target_pose"] = target_pose

    if len(speed_key) > 0:
        if data_config["direction_process"] == "midfwd":
            wind_pose = target_pose
        else:
            wind_pose = target_pose[window_inds]

        if data_config["direction_process"] in ["midfwd", "x360"]:
            wind_root = data["root"]
        else:
            wind_root = data["root"][window_inds]

        if ("part_speed" in speed_key) or ("avg_speed_3d" in speed_key):
            root_spd = torch.sqrt(
                (torch.diff(wind_root, n=1, axis=1) ** 2).sum(axis=-1)
            ).mean(axis=1)
            parts = [
                [0, 1, 2, 3, 4, 5],  # spine and head
                [1, 6, 7, 8, 9, 10, 11],  # arms from front spine
                [5, 12, 13, 14, 15, 16, 17],  # legs from back spine
            ]
            dxyz = torch.zeros((len(root_spd), 3))
            for i, part in enumerate(parts):
                pose_part = (
                    wind_pose
                    - wind_pose[:, window // 2, None, part[0] : part[0] + 1, :]
                )
                relative_dxyz = (
                    torch.diff(
                        pose_part[:, :, part[1:], :],
                        n=1,
                        axis=1,
                    )
                    ** 2
                ).sum(axis=-1)
                dxyz[:, i] = torch.sqrt(relative_dxyz).mean(axis=(1, 2))
            if "avg_speed_3d" in speed_key:
                speed = torch.cat(
                    [
                        root_spd[:, None],  # root
                        dxyz[:, 0:1],  # spine and head
                        dxyz[:, 1:].mean(axis=-1, keepdims=True),  # limbs
                    ],
                    axis=-1,
                )
        else:
            speed = np.diff(pose, n=1, axis=0, prepend=pose[0:1])
            speed = np.sqrt((speed**2).sum(axis=-1)).mean(axis=-1)
            speed = torch.tensor(speed[window_inds].mean(axis=-1, keepdims=True), dtype=torch.float32)

    if len(speed_key) > 0:
        data[speed_key[0]] = speed

    return data, window_inds
