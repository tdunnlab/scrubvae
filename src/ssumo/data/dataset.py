from neuroposelib import read, preprocess
import numpy as np
import ssumo.data.quaternion as qtn
from typing import Optional, Type, Union, List
from torch.utils.data import Dataset
import torch
from numpy.lib.stride_tricks import sliding_window_view
from tqdm import trange


def inv_kin_torch(
    pose,
    kinematic_tree,
    offset,
    forward_indices=[0, 1],
    device="cuda",
):
    """
    Adapted from T2M-GPT (https://mael-zys.github.io/T2M-GPT/)
    [1] Zhang, Jianrong, et al. "Generating Human Motion From Textual
    Descriptions With Discrete Representations." Proceedings of the
    IEEE/CVF Conference on Computer Vision and Pattern Recognition. 2023.
    """

    # Find forward root direction
    forward = pose[:, forward_indices[1], :] - pose[:, forward_indices[0], :]
    forward = (
        forward.type(torch.FloatTensor).to(device)
        / torch.norm(forward, dim=-1)[..., None]
    )

    # Root Rotation
    target = (
        torch.tensor([[1, 0, 0]])
        .repeat([len(forward), 1])
        .type(torch.FloatTensor)
        .to(device)
    )
    root_quat = qtn.qbetween(forward, target)
    root_quat[torch.norm(root_quat, dim=1) == 0] = torch.tensor(
        [0.0, 0.0, 0.0, 1.0]
    ).to(device)

    local_quat = torch.zeros(pose.shape[:-1] + (4,))
    root_quat[0] = torch.tensor([[1.0, 0.0, 0.0, 0.0]])
    local_quat[:, 0] = root_quat
    for chain in kinematic_tree:
        R = root_quat
        for i in range(len(chain) - 1):
            u = offset[chain[i + 1]][None, ...]
            u = u.repeat([len(pose)] + [1 for j in u.shape[1:]])
            v = pose[:, chain[i + 1]] - pose[:, chain[i]]
            v = v / torch.norm(v, dim=-1)[..., None]
            rot_u_v = qtn.qbetween(u, v)
            R_loc = qtn.qmul(qtn.qinv(R), rot_u_v)
            local_quat[:, chain[i + 1], :] = R_loc
            R = qtn.qmul(R, R_loc)

    return local_quat


def inv_kin(
    pose: np.ndarray,
    kinematic_tree: Union[List, np.ndarray],
    offset: np.ndarray,
    forward_indices: Union[List, np.ndarray] = [0, 1],
):
    """
    Adapted from T2M-GPT (https://mael-zys.github.io/T2M-GPT/)
    [1] Zhang, Jianrong, et al. "Generating Human Motion From Textual
    Descriptions With Discrete Representations." Proceedings of the
    IEEE/CVF Conference on Computer Vision and Pattern Recognition. 2023.
    """

    # Find forward root direction
    forward = pose[:, forward_indices[1], :] - pose[:, forward_indices[0], :]
    forward = forward / np.linalg.norm(forward, axis=-1)[..., None]

    # Root Rotation
    target = np.array([[1, 0, 0]]).repeat(len(forward), axis=0)
    root_quat = qtn.qbetween_np(forward, target)

    local_quat = np.zeros(pose.shape[:-1] + (4,))
    root_quat[0] = np.array([[1.0, 0.0, 0.0, 0.0]])
    local_quat[:, 0] = root_quat
    for chain in kinematic_tree:
        R = root_quat
        for i in range(len(chain) - 1):
            u = offset[chain[i + 1]][None, ...].repeat(len(pose), axis=0)
            v = pose[:, chain[i + 1]] - pose[:, chain[i]]
            v = v / np.linalg.norm(v, axis=-1)[..., None]
            rot_u_v = qtn.qbetween_np(u, v)
            R_loc = qtn.qmul_np(qtn.qinv_np(R), rot_u_v)
            local_quat[:, chain[i + 1], :] = R_loc
            R = qtn.qmul_np(R, R_loc)

    return local_quat


def fwd_kin_cont6d(
    continuous_6D: np.ndarray,
    kinematic_tree: Union[List, np.ndarray],
    offset: np.ndarray,
    root_pos: np.ndarray,
    do_root_R: bool = True,
):
    # continuous_6D (batch_size, pose_num, 6)
    # pose (batch_size, pose_num, 3)
    # root_pos (batch_size, 3)

    pose = np.zeros(continuous_6D.shape[:-1] + (3,))
    pose[:, 0] = root_pos

    if len(offset.shape) == 2:
        offsets = np.moveaxis(np.tile(offset[..., None], continuous_6D.shape[0]), -1, 0)
    else:
        offsets = offset

    for chain in kinematic_tree:
        if do_root_R:
            matR = qtn.cont6d_to_matrix_np(continuous_6D[:, 0])
        else:
            matR = np.eye(3)[np.newaxis, :].repeat(len(continuous_6D), axis=0)
        for i in range(1, len(chain)):
            matR = np.matmul(matR, qtn.cont6d_to_matrix_np(continuous_6D[:, chain[i]]))
            offset_vec = offsets[:, chain[i]][..., np.newaxis]
            # print(matR.shape, offset_vec.shape)
            pose[:, chain[i]] = (
                np.matmul(matR, offset_vec).squeeze(-1) + pose[:, chain[i - 1]]
            )
    return pose


def fwd_kin_cont6d_torch(
    continuous_6d, kinematic_tree, offset, root_pos, do_root_R=True, eps=0
):
    # continuous_6d (batch_size, joints_num, 6)
    # joints (batch_size, joints_num, 3)
    # root_pos (batch_size, 3)

    if len(offset.shape) == 2:
        offsets = offset.expand(continuous_6d.shape[0], -1, -1)
    else:
        offsets = offset

    pose = torch.zeros(continuous_6d.shape[:-1] + (3,), device=continuous_6d.device)
    pose[..., 0, :] = root_pos
    for chain in kinematic_tree:
        if do_root_R:
            matR = qtn.cont6d_to_matrix(continuous_6d[:, 0], eps=eps)
        else:
            matR = (
                torch.eye(3)
                .expand((len(continuous_6d), -1, -1))
                .detach()
                .to(continuous_6d.device)
            )
        for i in range(1, len(chain)):
            matR = torch.matmul(
                matR, qtn.cont6d_to_matrix(continuous_6d[:, chain[i]], eps=eps)
            )
            offset_vec = offsets[:, chain[i]].unsqueeze(-1)

            pose[:, chain[i]] = (
                torch.matmul(matR, offset_vec).squeeze(-1) + pose[:, chain[i - 1]]
            )
    return pose


def normalize_root(root, arena_size):
    norm_root = root - arena_size[0]
    norm_root = 2 * norm_root / (arena_size[1] - arena_size[0]) - 1
    return norm_root


def inv_normalize_root(norm_root, arena_size):
    root = 0.5 * (norm_root + 1) * (arena_size[1] - arena_size[0])
    root += arena_size[0]
    return root


def get_speed_parts(pose, parts):
    print("Getting speed by body parts")
    root_spd = np.diff(pose[:, 0, :], n=1, axis=0, prepend=pose[0:1, 0, :]) ** 2
    dxyz = np.zeros((len(root_spd), len(parts) + 1))
    dxyz[:, 0] = np.sqrt(root_spd).sum(axis=-1)

    centered_pose = preprocess.center_spine(pose, keypt_idx=0)
    # ego_pose = preprocess.rotate_spine(
    #     centered_pose,
    #     keypt_idx=[0, 1],
    #     lock_to_x=False,
    # )

    for i, part in enumerate(parts):
        if part[0] == 0:
            pose_part = centered_pose
        else:
            pose_part = centered_pose - centered_pose[:, part[0] : part[0] + 1, :]
        relative_dxyz = (
            np.diff(
                pose_part[:, part[1:], :],
                n=1,
                axis=0,
                prepend=pose_part[0:1, part[1:], :],
            )
            ** 2
        ).sum(axis=-1)
        dxyz[:, i + 1] = np.sqrt(relative_dxyz).mean(axis=-1)

    return dxyz


def get_window_indices(ids, stride, window):
    print("Calculating windowed indices ...")
    window_inds = []
    frame_idx = np.arange(len(ids), dtype=int)
    id_diff = np.diff(ids, prepend=ids[0])
    id_change = np.concatenate([[0], np.where(id_diff != 0)[0], [len(ids)]])
    for i in trange(0, len(id_change) - 1):
        strided_data = sliding_window_view(
            frame_idx[id_change[i] : id_change[i + 1], ...],
            window_shape=window,
            axis=0,
        )[::stride, ...]
        window_inds += [torch.squeeze(torch.tensor(strided_data, dtype=int))]
        assert (
            np.moveaxis(strided_data[1, ...], -1, 0)
            - frame_idx[id_change[i] + stride : id_change[i] + window + stride, ...]
        ).sum() == 0

    window_inds = torch.cat(window_inds, dim=0)

    return window_inds


def get_frame_yaw(pose, root_i=0, front_i=1):
    forward = pose[:, front_i, :] - pose[:, root_i, :]
    forward = forward / np.linalg.norm(forward, axis=-1)[..., None]
    yaw = -np.arctan2(forward[:, 1], forward[:, 0])
    return yaw


def get_heading2D(pose, root_i=0, front_i=1):
    yaw = get_frame_yaw(pose, root_i, front_i)
    heading2D = get_angle2D(yaw[:, None])
    heading_change = np.diff(heading2D, n=1, axis=0, prepend=heading2D[0:1, :])

    return np.append(heading2D, heading_change, axis=-1)


def get_angle2D(angle):  # sin is first, then cos
    angle2D = np.concatenate([np.sin(angle)[:, None], np.cos(angle)[:, None]], axis=-1)
    angle2D = angle2D.reshape(angle.shape[:-1] + (-1,))
    return angle2D


def get_angle_from_2D(angle2D):
    angle2D = angle2D.reshape(angle2D.shape[0], -1, 2)
    angles = np.arctan2(angle2D[..., 0], angle2D[..., 1])
    return angles


def get_segment_len(pose: np.ndarray, kinematic_tree: np.ndarray, offset: np.ndarray):
    parents = [0] * len(offset)
    parents[0] = -1
    for chain in kinematic_tree:
        for j in range(1, len(chain)):
            parents[chain[j]] = chain[j - 1]

    offsets = np.moveaxis(np.tile(offset[..., None], pose.shape[0]), -1, 0)
    for i in range(1, offset.shape[0]):
        offsets[:, i] = (
            np.linalg.norm(pose[:, i, :] - pose[:, parents[i], :], axis=1)[..., None]
            * offsets[:, i]
        )

    return offsets


def get_segment_len_torch(pose, kinematic_tree, offset, device):
    parents = [0] * len(offset)
    parents[0] = -1
    for chain in kinematic_tree:
        for j in range(1, len(chain)):
            parents[chain[j]] = chain[j - 1]

    offsets = torch.moveaxis(torch.tile(offset[..., None], (pose.shape[0],)), -1, 0).to(
        device
    )
    for i in range(1, offset.shape[0]):
        offsets[:, i] = (
            torch.norm(pose[:, i, :] - pose[:, parents[i], :], dim=1)[..., None]
            * offsets[:, i]
        )

    return offsets


def get_speed_outliers(pose, window_inds, threshold=2.25):
    avg_spd = np.diff(pose, n=1, axis=0, prepend=pose[0:1])
    avg_spd = np.sqrt((avg_spd**2).sum(axis=-1)).mean(axis=-1, keepdims=True)
    outlier_frames = np.where(
        avg_spd[window_inds[:, 1:], ...].mean(
            axis=tuple(range(1, len(avg_spd.shape) + 1))
        )
        > threshold
    )[0]
    outlier_frames = np.unique(outlier_frames)
    print("Outlier frames above {}: {}".format(threshold, len(outlier_frames)))
    return outlier_frames


class MouseDataset(Dataset):
    def __init__(
        self,
        data,
        window_inds,
        arena_size=None,
        kinematic_tree=None,
        n_keypts=None,
        label="Train",
        discrete_classes=None,
    ):
        self.data = data
        self.window_inds = window_inds
        self.n_keypts = n_keypts
        self.discrete_classes = discrete_classes

        if arena_size is not None:
            self.arena_size = torch.tensor(arena_size)
        else:
            self.arena_size = None

        self.kinematic_tree = kinematic_tree
        self.ind_with_window_inds = [
            k for k, v in self.data.items() if v.shape[0] != len(self.window_inds)
        ]
        self.label = label

    def __len__(self):
        return len(self.window_inds)

    def __getitem__(self, idx):
        query = {
            k: self.data[k][self.window_inds[idx]] for k in self.ind_with_window_inds
        }

        query.update(
            {
                k: v[idx]
                for k, v in self.data.items()
                if k not in self.ind_with_window_inds
            }
        )
        return query


class MouseDataset2D(MouseDataset):
    def __getitem__(self, idx):
        query = {
            k: self.data[k][self.window_inds[idx]]
            for k in self.ind_with_window_inds
            if k != "raw_pose"
        }

        if "raw_pose" in self.ind_with_window_inds:
            query["raw_pose"] = self.data["raw_pose"][
                torch.remainder(self.window_inds[idx], len(self.data["raw_pose"]))
            ]

        query.update(
            {
                k: v[idx]
                for k, v in self.data.items()
                if k not in self.ind_with_window_inds
            }
        )
        return query
