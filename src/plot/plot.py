import numpy as np
from dappy import DataStruct as ds
from typing import Union, List
from dappy import visualization as vis
import matplotlib.pyplot as plt
from constants import PLANE


# def keypoints(
#     pose: np.ndarray,
#     frames,
#     keypoints: List,
#     joint_colors: np.ndarray,
#     filepath: str = "./",
# ):
#     axis = ["x", "y", "z"]
#     f, ax = plt.subplots(len(keypoints) * pose.shape[-1], 1, figsize=(10, 3))
#     for i, kpts in enumerate(keypoints):
#         for j in range(3):
#             # import pdb; pdb.set_trace()
#             ax[i * 3 + j].plot(
#                 np.arange(len(frames)),
#                 pose[frames, kpts, j],
#                 c=joint_colors[i],
#                 alpha=1,
#                 lw=0.5,
#             )
#             ax[i * 3 + j].axis("off")
#             # ax[i*3 + j].set_ylabel(axis[j])

#     f.subplots_adjust(hspace=-0.25)
#     plt.savefig("{}/vis_kptrace.png".format(filepath), dpi=400)
#     plt.close()

#     return


# from dappy import read, preprocess

# skeleton_config = read.config(
#     "/mnt/home/jwu10/working/behavior_vae/configs/mouse_skeleton.yaml"
# )
# REORDER = [4, 3, 2, 1, 0, 5, 11, 10, 9, 8, 7, 6, 17, 16, 15, 14, 13, 12]
# pose = read.pose_h5(
#     "/mnt/home/jwu10/working/ceph/data/ensemble_healthy/pose_aligned.h5"
# )[0][:, REORDER, :]
# pose = preprocess.rotate_spine(
#     preprocess.center_spine(pose, keypt_idx=0),
#     keypt_idx=[0, 1],
#     lock_to_x=False,
# )

# colors = np.array(skeleton_config["KEYPT_COLORS"])
# ryb = np.array([[1, 0.1, 0.1], "#F6BE00", [0.1, 0.1, 1]], dtype=object)
# kpts = [5, 8, 15]
# keypoints(
#     pose,
#     np.arange(1000, 6000, 2).astype(int),
#     keypoints=kpts,
#     joint_colors=ryb,
#     filepath="./",
# )


def trace(
    pose: np.ndarray,
    connectivity: ds.Connectivity,
    vis_plane: str = "xz",
    frames: Union[List[int], int] = [3000, 100000, 500000],
    n_full_pose: int = 3,
    keypts_to_trace: List[int] = [0, 4, 8, 11, 14, 17],
    centered: bool = True,
    N_FRAMES: int = 300,
    dpi: int = 200,
    FIG_NAME: str = "pose_trace.png",
    SAVE_ROOT: str = "./test/pose_vids/",
):
    if isinstance(frames, int):
        frames = [frames]

    figsize = (10, 10)
    fig = plt.figure(figsize=figsize)
    ax = fig.add_subplot(1, 1, 1)
    pose_vis, limits, links, COLORS = vis.pose._init_vid3D(
        pose, connectivity, np.array(frames, dtype=int), centered, N_FRAMES, SAVE_ROOT
    )

    plane_idx = [PLANE[k] for k in vis_plane]
    pose_vis = pose_vis[..., plane_idx]

    full_pose_inds = np.linspace(0, N_FRAMES - 1, n_full_pose).astype(int)
    for i in full_pose_inds:
        curr_frames = i + np.arange(len(frames)) * N_FRAMES

        for index_from, index_to in connectivity.links:
            xs, ys = [
                np.array(
                    [
                        pose_vis[curr_frames, index_from, j],
                        pose_vis[curr_frames, index_to, j],
                    ]
                )
                for j in range(2)
            ]
            lw_color = np.sqrt(np.linspace(0, 0.75, 10))
            linewidth = 3.5 - np.linspace(0, 3.1, 10)
            for co, l in zip(lw_color, linewidth):
                ax.plot(
                    xs,
                    ys,
                    c=(co, co, co),
                    lw=l,
                    alpha=0.2,  # - (i * 0.55 / full_pose_inds[-1])
                )

        ax.scatter(
            pose_vis[curr_frames, :, 0].flatten(),
            pose_vis[curr_frames, :, 1].flatten(),
            marker="o",
            color=np.tile(connectivity.keypt_colors, (len(frames), 1)),
            s=75,
            alpha=0.3,  # 1 - (i * 0.75 / full_pose_inds[-1]),
            zorder=3.5,
        )

    # trace_frames = np.arange(N_FRAMES)
    # trace_frames = trace_frames[~np.isin(trace_frames, full_pose_inds)]

    for keypt in keypts_to_trace:
        ax.plot(
            pose_vis[:, keypt, 0].reshape(len(frames), -1).T,
            pose_vis[:, keypt, 1].reshape(len(frames), -1).T,
            marker="o",
            color=connectivity.keypt_colors[keypt],
            ms=0,
            lw=2.5,
            alpha=0.3,
        )
    ax.set_aspect("equal")
    ax.axis("off")
    plt.savefig("{}/vis{}_{}".format(SAVE_ROOT, vis_plane, FIG_NAME), dpi=dpi)
    plt.close()

    return
