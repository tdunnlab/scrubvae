from dappy import read, preprocess
import numpy as np
import ssumo.data.quaternion as qtn
from typing import List
import torch
from ssumo.data.dataset import *
from torch.utils.data import DataLoader


def babel_data(
    data_config: dict,
    window: int = 51,
    train: bool = True,
    data_keys: List[str] = ["x6d", "root", "offsets"],
    shuffle: bool = False,
    recompute: bool = True,
    save: bool = False,
):
    """Read in BABEL dataset
    Punnakkal, Abhinanda R, et al
    "BABEL: Bodies, Action and Behavior with English Labels",
    Proceedings IEEE/CVF Conf.~on Computer Vision and Pattern Recognition (CVPR), 2021
    https://babel.is.tue.mpg.de/

    Args:
        data_config (dict): Dict of parameter options for loading in data.
        window (int, optional): # of frames per sample to be given by DataLoader class. Defaults to 51.
        train (bool, optional): Read in train or test set. Defaults to True.
        data_keys (List[str], optional): Data fields to be given by DataLoader class. Defaults to ["x6d", "root", "offsets"].
        shuffle (bool, optional): Whether DataLoader shuffles. Defaults to False.
        recompute (bool, optional): Whether to recompute inverse kinematics and more, or load a preprocessed file. Defaults to True.
        save (bool, optional): Whether to save computed inverse kinematics and more. Defaults to False.

    """

    skeleton_config = read.config(data_config["skeleton_path"])

    pose, ids = read.pose_h5(data_config["data_path"])

    vidlen = np.unique(ids[1], return_counts=True)[1]
    vidlenfilter = np.argwhere(vidlen >= window + data_config["stride"])[:, 0]
    vidlenfilter = np.in1d(ids[1], vidlenfilter)

    ids = ids[:, vidlenfilter]
    pose = pose[vidlenfilter]

    pose = pose[ids[0] == int(train)]
    ids = ids[1][ids[0] == int(train)]

    pose = pose[:, [3, 1, 2, 0] + [i for i in range(4, 22)] + [24, 36, 39, 51]]

    # Load or index train or test
    if data_config["filter_pose"]:
        pose = preprocess.median_filter(pose, ids, 5)

    # Save raw pose in dataset if specified
    data = {"raw_pose": pose} if "raw_pose" in data_keys else {}

    # Get windowed indices (n_frames x window)
    window_inds = get_window_indices(ids, data_config["stride"], window)

    windowed_yaw = get_frame_yaw(pose, [0, 2, 1])[window_inds]

    yaw = windowed_yaw[:, window // 2][..., None]

    if "heading" in data_keys:
        data["heading"] = get_angle2D(yaw)

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
        if recompute:
            print("Applying inverse kinematics ...")
            local_qtn = inv_kin(
                pose,
                skeleton_config["KINEMATIC_TREE"],
                np.array(skeleton_config["OFFSET"]),
                forward_indices=[0, 2, 1],
            )
            if save:
                np.save(
                    data_config["data_path"][: data_config["data_path"].rfind("/")]
                    + "/babel_invkin"
                    + ["test", "train"][train]
                    + ".npy",
                    local_qtn,
                    allow_pickle=True,
                )
        else:
            print("Loading inverse kinematics ...")
            local_qtn = np.load(
                data_config["data_path"][: data_config["data_path"].rfind("/")]
                + "/babel_invkin"
                + ["test", "train"][train]
                + ".npy",
                allow_pickle=True,
            )
        if "midfwd" in data_config["direction_process"]:
            ## Center frame of a window is translated to center and rotated to x+
            fwd_qtn = np.zeros((len(window_inds), 4))
            fwd_qtn[:, [-1, 0]] = get_angle2D(yaw / 2)

            local_qtn = local_qtn[window_inds]
            fwd_qtn = np.repeat(fwd_qtn[:, None, :], window, axis=1)

            local_qtn[..., 0, :] = qtn.qmul_np(fwd_qtn, local_qtn[..., 0, :])

            if "root" in data_keys:
                root = qtn.qrot_np(fwd_qtn, root)

            assert len(root) == len(window_inds)
            assert len(local_qtn) == len(window_inds)

        data["x6d"] = qtn.quaternion_to_cont6d_np(local_qtn)

    # Scale offsets by segment lengths
    if "offsets" in data_keys:
        if recompute:
            data["offsets"] = get_segment_len(
                pose,
                skeleton_config["KINEMATIC_TREE"],
                np.array(skeleton_config["OFFSET"]),
            )
            if save:
                np.save(
                    data_config["data_path"][: data_config["data_path"].rfind("/")]
                    + "/babel_offsets"
                    + ["test", "train"][train]
                    + ".npy",
                    data["offsets"],
                    allow_pickle=True,
                )
        else:
            data["offsets"] = np.load(
                data_config["data_path"][: data_config["data_path"].rfind("/")]
                + "/babel_offsets"
                + ["test", "train"][train]
                + ".npy",
                allow_pickle=True,
            )

    if "root" in data_keys:
        data["root"] = root

    # Move everything to tensors
    data = {k: torch.tensor(v, dtype=torch.float32) for k, v in data.items()}

    if "target_pose" in data_keys:
        if recompute:
            data["target_pose"] = fwd_kin_cont6d_torch(
                data["x6d"],
                skeleton_config["KINEMATIC_TREE"],
                data["offsets"],
                root_pos=torch.zeros(data["x6d"].shape[0], 3),
                do_root_R=True,
                eps=1e-8,
            )
            if save:
                np.save(
                    data_config["data_path"][: data_config["data_path"].rfind("/")]
                    + "/babel_targetpose"
                    + ["test", "train"][train]
                    + ".npy",
                    data["target_pose"],
                    allow_pickle=True,
                )
        else:
            data["target_pose"] = np.load(
                data_config["data_path"][: data_config["data_path"].rfind("/")]
                + "/babel_targetpose"
                + ["test", "train"][train]
                + ".npy",
                allow_pickle=True,
            )

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


def mouse_data(
    data_config: dict,
    window: int = 51,
    train_ids: List = [0, 1, 2],
    train: bool = True,
    data_keys: List[str] = ["x6d", "root", "offsets"],
    shuffle: bool = False,
    normalize: List[str] = [],
):
    REORDER = [4, 3, 2, 1, 0, 5, 11, 10, 9, 8, 7, 6, 17, 16, 15, 14, 13, 12]
    skeleton_config = read.config(data_config["skeleton_path"])
    pose, ids = read.pose_h5(data_config["data_path"], dtype=np.float64)

    set_ids = np.in1d(ids, train_ids) if train else ~np.in1d(ids, train_ids)
    pose = pose[set_ids][:, REORDER, :]

    ## Smoothing
    if data_config["filter_pose"]:
        pose = preprocess.median_filter(pose, ids[set_ids], 5)

    data = {"raw_pose": pose} if "raw_pose" in data_keys else {}
    window_inds = get_window_indices(ids[set_ids], data_config["stride"], window)

    speed_key = [key for key in data_keys if "speed" in key]
    assert len(speed_key) < 2
    if (len(speed_key) > 0) or (data_config["remove_speed_outliers"] is not None):
        if ("part_speed" in speed_key) or ("avg_speed_3d" in speed_key):
            speed = get_speed_parts(
                pose=pose,
                parts=[
                    [0, 1, 2, 3, 4, 5],  # spine and head
                    [1, 6, 7, 8, 9, 10, 11],  # arms from front spine
                    [5, 12, 13, 14, 15, 16, 17],  # left legs from back spine
                ],
            )
            if "avg_speed_3d" in speed_key:
                speed = np.concatenate(
                    [speed[:, :2], speed[:, 2:].mean(axis=-1, keepdims=True)], axis=-1
                )

        else:
            speed = np.diff(pose, n=1, axis=0, prepend=pose[0:1])
            speed = np.sqrt((speed**2).sum(axis=-1)).mean(axis=-1, keepdims=True)

    if data_config["remove_speed_outliers"] is not None:
        outlier_frames = np.where(
            speed[window_inds[:, 1:], ...].mean(
                axis=tuple(range(1, len(speed.shape) + 1))
            )
            > data_config["remove_speed_outliers"]
        )[0]
        outlier_frames = np.unique(outlier_frames)
        print(
            "Outlier frames above {}: {}".format(
                data_config["remove_speed_outliers"], len(outlier_frames)
            )
        )
        window_inds = np.delete(window_inds, outlier_frames, 0)

    if len(speed_key) > 0:
        data[speed_key[0]] = speed[window_inds[:, 1:]].mean(axis=1)

    windowed_yaw = get_frame_yaw(pose, 0, 1)[window_inds]

    if "heading_change" in data_keys:
        data["heading_change"] = np.diff(windowed_yaw, n=1, axis=-1).sum(
            axis=-1, keepdims=True
        )

    yaw = windowed_yaw[:, window // 2][..., None]

    if "heading" in data_keys:
        data["heading"] = get_angle2D(yaw)

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
            ## Center frame of a window is translated to center and rotated to x+
            fwd_qtn = np.zeros((len(window_inds), 4))
            fwd_qtn[:, [-1, 0]] = get_angle2D(yaw / 2)

            if "mid" in data_config["direction_process"]:
                local_qtn = local_qtn[window_inds]
                fwd_qtn = np.repeat(fwd_qtn[:, None, :], window, axis=1)

            local_qtn[..., 0, :] = qtn.qmul_np(fwd_qtn, local_qtn[..., 0, :])

            if "root" in data_keys:
                root = qtn.qrot_np(fwd_qtn, root)

            assert len(root) == len(window_inds)
            assert len(local_qtn) == len(window_inds)

        data["x6d"] = qtn.quaternion_to_cont6d_np(local_qtn)

    if "offsets" in data_keys:
        data["offsets"] = get_segment_len(
            pose,
            skeleton_config["KINEMATIC_TREE"],
            np.array(skeleton_config["OFFSET"]),
        )

    if "root" in data_keys:
        data["root"] = root

    data = {k: torch.tensor(v, dtype=torch.float32) for k, v in data.items()}

    for key in normalize:
        if (key != "heading") and (key in data_keys):
            if data_config["normalize"] == "bounded":
                print(
                    "Rescaling decoding variable, {}, to be between -1 and 1".format(
                        key
                    )
                )
                key_min = data[key].min(dim=0)[0] - data[key].min(dim=0)[0] * 0.1
                data[key] -= key_min
                key_max = data[key].max(dim=0)[0] + data[key].max(dim=0)[0] * 0.1
                data[key] = 2 * data[key] / key_max - 1
                assert data[key].max() < 1
                assert data[key].min() > -1
            elif data_config["normalize"] == "z_score":
                print(
                    "Mean centering and unit standard deviation-scaling {}".format(key)
                )
                data[key] -= data[key].mean(axis=0)
                data[key] /= data[key].std(axis=0)

    if "target_pose" in data_keys:
        reshaped_x6d = data["x6d"].reshape((-1,) + data["x6d"].shape[-2:])
        if data_config["direction_process"] == "midfwd":
            offsets = data["offsets"][window_inds].reshape(
                reshaped_x6d.shape[:2] + (-1,)
            )
        else:
            offsets = data["offsets"]

        data["target_pose"] = fwd_kin_cont6d_torch(
            reshaped_x6d,
            skeleton_config["KINEMATIC_TREE"],
            offsets,
            root_pos=torch.zeros(reshaped_x6d.shape[0], 3),
            do_root_R=True,
            eps=1e-8,
        ).reshape(data["x6d"].shape[:-1] + (3,))

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
