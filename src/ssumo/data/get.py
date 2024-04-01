from dappy import read, preprocess
import numpy as np
import ssumo.data.quaternion as qtn
from typing import List
import torch
from ssumo.data.dataset import *
from torch.utils.data import DataLoader


def get_mouse(
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
        if data_config["normalize"] == "bounded":
            if key == "heading":
                pass
            else:
                print(
                    "Rescaling decoding variable, {}, to be between -1 and 1".format(
                        key
                    )
                )
                key_min = data[key].min(dim=0)[0] - data[key].min(dim=0)[0]*0.1
                # data[key] -= data[key].mean(axis=0)
                # data[key] /= data[key].std(axis=0)
                data[key] -= key_min
                key_max = data[key].max(dim=0)[0] + data[key].max(dim=0)[0]*0.1
                data[key] = 2 * data[key] / key_max - 1
                assert data[key].max() < 1
                assert data[key].min() > -1
            # norm_root = 2 * norm_root / (self.arena_size[1] - self.arena_size[0]) - 1
        elif data_config["normalize"] == "z_score":
            print("Mean centering and unit standard deviation-scaling {}".format(key))
            data[key] -= data[key].mean(axis=0)
            data[key] /= data[key].std(axis=0)


    if "target_pose" in data_keys:
        reshaped_x6d = data["x6d"].reshape((-1,) + data["x6d"].shape[-2:])
        if data_config["direction_process"]=="midfwd":
            offsets = data["offsets"][window_inds].reshape(reshaped_x6d.shape[:2] + (-1,))
        else:
            offsets = data["offsets"]
        data["target_pose"] = fwd_kin_cont6d_torch( reshaped_x6d, skeleton_config["KINEMATIC_TREE"], offsets, root_pos=torch.zeros(reshaped_x6d.shape[0], 3), do_root_R=True, eps=1e-8, ).reshape(data["x6d"].shape[:-1] + (3,))

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
