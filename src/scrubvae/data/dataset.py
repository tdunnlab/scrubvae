from neuroposelib import read
import numpy as np
import scrubvae.data.quaternion as qtn
from typing import Optional, Type, Union, List
from torch.utils.data import Dataset
import torch
from numpy.lib.stride_tricks import sliding_window_view
from tqdm import trange


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
    """
    Normalize root Cartesian coordinates to be from (-1, 1) based on arena size
    """
    norm_root = root - arena_size[0]
    norm_root = 2 * norm_root / (arena_size[1] - arena_size[0]) - 1
    return norm_root


def inv_normalize_root(norm_root, arena_size):
    root = 0.5 * (norm_root + 1) * (arena_size[1] - arena_size[0])
    root += arena_size[0]
    return root


def get_speed_parts(pose, parts):
    """
    Get the (1) average root displacement,
    (2) average speed of the spine relative to the root,
    and (3) average speed of the limbs relative to the spine
    """
    print("Getting speed by body parts")
    root_spd = np.diff(pose[..., 0, :], n=1, axis=-2) ** 2  # (n_samples, window, 3)
    root_spd = np.sqrt(root_spd.sum(-1)).mean(-1)
    dxyz = np.zeros((len(root_spd), len(parts) + 1))
    dxyz[:, 0] = root_spd  # TODO: Put sum in sqrt

    centered_pose = pose - pose[..., 0:1, :]
    for i, part in enumerate(parts):
        if part[0] == 0:
            pose_part = centered_pose
        else:
            pose_part = centered_pose - centered_pose[:, part[0] : part[0] + 1, :]

        relative_dxyz = (
            np.diff(
                pose_part[..., part[1:], :],
                n=1,
                axis=-3,
            )
            ** 2
        ).sum(-1)
        dxyz[:, i + 1] = np.sqrt(relative_dxyz).mean(axis=(-1, -2))

    return dxyz


def get_speed_parts_torch(pose, parts):
    """
    Pytorch version
    Get the (1) average root displacement,
    (2) average speed of the spine relative to the root,
    and (3) average speed of the limbs relative to the spine
    """
    print("Getting speed by body parts")
    root_spd = (
        torch.diff(pose[..., 0, :], n=1, dim=-3, prepend=pose[..., 0:1, 0, :]) ** 2
    )
    dxyz = torch.zeros((len(root_spd), len(parts) + 1), device=pose.device)
    dxyz[:, 0] = torch.sqrt(root_spd.sum(dim=-1))

    centered_pose = pose - pose[:, 0:1, :]

    for i, part in enumerate(parts):
        pose_part = centered_pose - centered_pose[:, part[0] : part[0] + 1, :]
        relative_dxyz = (
            torch.diff(
                pose_part[:, part[1:], :],
                n=1,
                dim=0,
                prepend=pose_part[0:1, part[1:], :],
            )
            ** 2
        ).sum(dim=-1)
        dxyz[:, i + 1] = torch.sqrt(relative_dxyz).mean(dim=-1)

    return dxyz


def get_window_indices(ids, stride, window):
    """
    Get full indices of an array broken up by sliding windows
    """
    print("Calculating windowed indices ...")
    window_inds = []
    frame_idx = np.arange(len(ids), dtype=int)
    id_diff = np.diff(ids, prepend=ids[0])
    id_change = np.concatenate([[0], np.where(id_diff != 0)[0], [len(ids)]])
    for i in trange(0, len(id_change) - 1):
        if (id_change[i + 1] - id_change[i]) >= window:
            strided_data = sliding_window_view(
                frame_idx[id_change[i] : id_change[i + 1], ...],
                window_shape=window,
                axis=0,
            )[::stride, ...]

            window_inds += [torch.tensor(strided_data, dtype=int)]

            if strided_data.shape[0] > 1:
                assert (
                    np.moveaxis(strided_data[1, ...], -1, 0)
                    - frame_idx[
                        id_change[i] + stride : id_change[i] + window + stride, ...
                    ]
                ).sum() == 0
        else:
            print(
                "ID {} length smaller than window size - skipping ...".format(
                    ids[id_change[i]]
                )
            )

    window_inds = torch.cat(window_inds, dim=0)

    return window_inds


def get_frame_yaw(pose, root_i=0, front_i=1):
    """
    Get yaw of given segment in radians
    """
    forward = pose[:, front_i, :] - pose[:, root_i, :]
    forward = forward / np.linalg.norm(forward, axis=-1)[..., None]
    yaw = -np.arctan2(forward[:, 1], forward[:, 0])
    return yaw


def get_heading2D(pose, root_i=0, front_i=1):
    """
    NOT USED

    Get yaw of given segment as [sin(angle), cos(angle)]
    i.e., coordinates on a unit circle
    """
    yaw = get_frame_yaw(pose, root_i, front_i)
    heading2D = get_angle2D(yaw[:, None])
    heading_change = np.diff(heading2D, n=1, axis=0, prepend=heading2D[0:1, :])

    return np.append(heading2D, heading_change, axis=-1)


def get_angle2D(angle):  # sin is first, then cos
    """
    Given angles in radians, return [sin(angle), cos(angle)]
    i.e., coordinates on a unit circle
    """
    angle2D = np.concatenate([np.sin(angle)[:, None], np.cos(angle)[:, None]], axis=-1)
    angle2D = angle2D.reshape(angle.shape[:-1] + (-1,))
    return angle2D


def get_angle_from_2D(angle2D):
    """
    Given coordinates on a unit circle, return angle in radians
    """
    angle2D = angle2D.reshape(angle2D.shape[0], -1, 2)
    angles = np.arctan2(angle2D[..., 0], angle2D[..., 1])
    return angles


def get_segment_len(pose: np.ndarray, kinematic_tree: np.ndarray, offset: np.ndarray):
    """
    Get length of all segments in a pose defined by a kinematic tree
    """
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


def get_speed_outliers(pose, threshold=2.25):
    """
    Find indices of frames in which the average speed is greater than the defined threshold
    """
    inds = np.arange(len(pose))

    avg_spd = np.diff(pose, n=1, axis=-3)
    avg_spd = np.sqrt((avg_spd**2).sum(axis=-1)).mean(axis=(-1, -2))
    outlier_frames = np.where(avg_spd > threshold)[0]
    print("Outlier frames above {}: {}".format(threshold, len(outlier_frames)))
    return outlier_frames



def preprocess_save_data(
    data_path: str,
    skeleton_config: dict,
    dataset: str,
    window: int,
    stride: int = 2,
    data_keys: List[str] = ["x6d", "root", "offsets"],
    speed_threshold: Optional[float] = 2.25,
    direction_process: str = "midfwd",
):
    """Prepare and save all data preprocessing for SC-VAE model training, validation, and testing

    Parameters
    ----------
    data_path : str
        Path to folder with datasets
    skeleton_config : dict
        Configuration file denoting the structure of the skeleton
    dataset : str
        Which dataset to prepare (i.e., "4_mice", "parkinsons")
    data_keys : List[str], optional
        Keys of data to save, by default ["x6d", "root", "offsets"]
    speed_threshold : Optional[float], optional
        Action segments with greater average speed will be filtered out, by default 2.25
    direction_process : str, optional
        Preprocess pose sequences such that the animals pass through the origin at the middle frame from
        any direction ("x360") or only in the x+ direction ("midfwd"), by default "midfwd"

    Returns
    -------
    data
        Dictionary with key-value pairs associated with `data_keys`
    """
    n_ids = 72 if dataset == "parkinsons" else 4
    print("Calculating dataset: {}".format(dataset))
    pose, ids = read.pose_h5("{}{}/pose.h5".format(data_path, dataset))
    window_inds = get_window_indices(ids, stride, window)
    pose = pose[window_inds]
    ids = ids[window_inds][:, window//2]

    # Filter out bad tracking using speed threshold
    if speed_threshold is not None:
        outlier_frames = get_speed_outliers(pose, speed_threshold)
        pose = np.delete(pose, outlier_frames, 0)
        ids = np.delete(ids, outlier_frames, 0)

    data_len = len(pose)
    data = {"raw_pose": pose}
    # Calculate the speed representation
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

    # Get yaw of mid-spine -> fwd-spine segment for central frame in all windows
    yaw = get_frame_yaw(pose[:, window // 2, ...], 0, 1)[..., None]

    # Convert to a 2D representation using the sin and cos of the yaw
    if "heading" in data_keys:
        data["heading"] = get_angle2D(yaw)

    if ("root" or "x6d") in data_keys:
        root = pose[..., 0, :].copy()  # (n_samples, window, 3)
        if direction_process in ["midfwd", "x360"]:
            # Centering root of middle frame
            root_center = np.zeros(root.shape)  # (n_samples, window, 3)
            root_center[..., [0, 1]] = root[:, window // 2, [0, 1]][
                :, None, :
            ]  # (n_samples, 1, 2)

            root -= root_center

    # Applying inverse kinematics to get local quaternions
    # That representation is then converted to a continuous 6D rotation
    if "x6d" in data_keys:
        print("Applying inverse kinematics ...")
        local_qtn = inv_kin(
            pose.reshape((-1,) + pose.shape[-2:]),
            skeleton_config["KINEMATIC_TREE"],
            np.array(skeleton_config["OFFSET"]),
            forward_indices=[1, 0],
        ).reshape(pose.shape[:-1] + (-1,))

        if direction_process == "midfwd":
            ## Center frame of a window is translated to center and rotated to x+
            fwd_qtn = np.zeros((len(yaw), 4))
            fwd_qtn[:, [-1, 0]] = get_angle2D(yaw / 2)
            fwd_qtn = np.repeat(fwd_qtn[:, None, :], window, axis=1)
            local_qtn[..., 0, :] = qtn.qmul_np(fwd_qtn, local_qtn[..., 0, :])

            if "root" in data_keys:
                root = qtn.qrot_np(fwd_qtn, root)

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
    if "ids" in data_keys:
        data["ids"] = torch.tensor(ids, dtype=torch.int16)

    if "target_pose" in data_keys:
        # Target pose root does not move
        reshaped_x6d = data["x6d"].reshape((-1,) + data["x6d"].shape[-2:])
        offsets = data["offsets"].reshape(reshaped_x6d.shape[:2] + (-1,))
        data["target_pose"] = fwd_kin_cont6d_torch(
            reshaped_x6d,
            skeleton_config["KINEMATIC_TREE"],
            offsets,
            root_pos=torch.zeros(reshaped_x6d.shape[0], 3),
            do_root_R=True,
            eps=1e-8,
        ).reshape(data["x6d"].shape[:-1] + (3,))

    for k, v in data.items():
        assert len(v) == data_len

    return data

class MouseDataset(Dataset):
    """
    Dataset class for Mouse dataset
    """

    def __init__(
        self,
        data,
        arena_size=None,
        kinematic_tree=None,
        n_keypts=None,
        label="train",
        discrete_classes=None,
        norm_params=None,
    ):
        self.data_keys = list(data.keys())
        self.data = data
        self.n_keypts = n_keypts
        self.discrete_classes = discrete_classes
        self.norm_params = norm_params

        if arena_size is not None:
            self.arena_size = torch.tensor(arena_size)
        else:
            self.arena_size = None

        self.kinematic_tree = kinematic_tree

        # # List of items which have already been windowed
        # self.ind_with_window_inds = [
        #     k for k, v in self.data.items() if v.shape[0] != len(self.window_inds)
        # ]
        self.label = label

    def __len__(self):
        return len(self.data[self.data_keys[0]])

    def __getitem__(self, idx):
        # Use window indices to access arrays which have not been windowed
        # query = {
        #     k: self.data[k][self.window_inds[idx]] for k in self.ind_with_window_inds
        # }

        # Query items which have already been windowed
        query = {
            k: v[idx]
            for k, v in self.data.items()
            # if k not in self.ind_with_window_inds
        }
        return query
