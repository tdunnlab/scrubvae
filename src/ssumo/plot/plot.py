import numpy as np
from dappy import DataStruct as ds
from typing import Union, List
from dappy import visualization as vis
import matplotlib.pyplot as plt
from ssumo.plot.constants import PLANE
from pathlib import Path
from matplotlib.lines import Line2D
from typing import Union, List, Optional, Dict, Tuple
import pandas as pd
import seaborn as sns
import colorcet as cc

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


def sample_clusters(pose, k_pred, connectivity, save_root):
    window = pose.shape[1]
    n_keypts = pose.shape[2]
    pose = pose - pose[:, window // 2 : window // 2 + 1, 0:1, :]
    ### Sample 9 videos from each cluster
    n_samples = 9
    indices = np.arange(len(k_pred))
    assert len(pose) == len(k_pred)
    for cluster in np.unique(k_pred):
        label_idx = indices[k_pred == cluster]
        num_points = min(len(label_idx), n_samples)
        permuted_points = np.random.permutation(label_idx)
        sampled_points = []
        for i in range(len(permuted_points)):
            if len(sampled_points) == num_points:
                break
            elif any(np.abs(permuted_points[i] - np.array(sampled_points)) < 100):
                continue
            else:
                sampled_points += [permuted_points[i]]

        print("Plotting Poses from Cluster {}".format(cluster))
        print(sampled_points)

        num_points = len(sampled_points)

        raw_pose = pose[sampled_points, ...].reshape(-1, n_keypts, pose.shape[-1])

        if num_points == n_samples:
            n_trans = 100
            plot_trans = (
                np.array(
                    [
                        [0, 0],
                        [1, 1],
                        [1, -1],
                        [-1, 1],
                        [-1, -1],
                        [1.5, 0],
                        [0, 1.5],
                        [-1.5, 0],
                        [0, -1.5],
                    ]
                )
                * n_trans
            )
            # plot_trans = np.append(plot_trans, np.zeros(n_samples)[:, None], axis=-1)
            raw_pose += np.repeat(plot_trans, window, axis=0)[:, None, :]
        # raw_pose = dataset[sampled_points]["raw_pose"].reshape(
        #     num_points * config["window"], dataset.n_keypts, 3
        # )

        vis.pose.arena3D(
            raw_pose,
            connectivity,
            frames=np.arange(num_points) * window,
            centered=False,
            N_FRAMES=window,
            fps=30,
            dpi=200,
            VID_NAME="cluster{}.mp4".format(cluster),
            SAVE_ROOT=save_root,
        )


def feature_ridge(
    feature: np.ndarray,
    labels: Union[List, np.ndarray],
    xlabel: str,
    ylabel: str,
    n_bins: int = 100,
    row_order: Optional[List] = None,
    binrange: Optional[List] = None,
    x_lim: Optional[List] = None,
    xticks: Optional[List] = None,
    medians: bool = False,
    path: str = "./",
):
    sns.set_theme(
        style="white", rc={"axes.facecolor": (0, 0, 0, 0), "figure.figsize": (20, 20)}
    )
    df = pd.DataFrame({"x": feature, "y": labels})
    height = 0.75
    pal = sns.cubehelix_palette(len(np.unique(labels)), rot=-0.25, light=0.7)
    grid = sns.FacetGrid(
        df, row="y", hue="y", aspect=20, row_order=row_order, height=height, palette=pal
    )

    grid.map(
        sns.histplot,
        "x",
        stat="probability",
        bins=n_bins,
        binrange=binrange,
        common_norm=False,
        common_bins=True,
        fill=True,
        alpha=0.5,
    )

    def adjust_xlim(x, color, label):
        ax = plt.gca()
        ax.set_xlim(left=x_lim[0], right=x_lim[1])

    grid.refline(y=0, linewidth=1, linestyle="-", color=None, clip_on=False)

    # Define and use a simple function to label the plot in axes coordinates
    def labelx(x, color, label):
        ax = plt.gca()
        ax.text(
            0,
            0.2,
            label,
            fontweight="bold",
            color=color,
            ha="left",
            va="center",
            transform=ax.transAxes,
        )

    grid.map(labelx, "x")
    if x_lim is not None:
        grid.map(adjust_xlim, "x")

    # Set the subplots to overlap
    grid.figure.subplots_adjust(hspace=-0.25)

    # Remove axes details that don't play well with overlap
    grid.set_titles("")
    grid.set(yticks=[], ylabel="")
    if xticks is not None:
        grid.set(xticks=xticks[0])
        grid.set_xticklabels(xticks[1])
    grid.despine(bottom=True, left=True)
    plt.xlabel(xlabel)
    grid.fig.text(0.03, 0.4, ylabel, rotation="vertical")

    if medians:
        ax_keys = list(grid.axes_dict.keys())
        for i, key in enumerate(ax_keys):
            ax = grid.axes_dict[key]
            for j in range(i, len(ax_keys)):
                ax.axvline(
                    np.median(df["x"].loc[df["y"] == ax_keys[j]]),
                    color=pal[j],
                    linewidth=3.5,
                    linestyle="-",
                )

            if i == 0:
                handles = [
                    Line2D(
                        [0],
                        [0],
                        linestyle="-",
                        color="k",
                        linewidth=3.5,
                        label="Median",
                    )
                ]
                ax.legend(handles, ["Median"], facecolor="white")

    # Path(path).mkdir(parents=True, exist_ok=True)
    plt.savefig(path + "ridge.png".format(xlabel, ylabel))
    plt.close()


def scatter_cmap(data, hue, label, path, cmap="cyclic"):
    if cmap == "cyclic":
        cmap = cc.cm["colorwheel"]

    plt.scatter(
        data[:, 0],
        data[:, 1],
        marker=".",
        s=0.4,
        c=hue,
        cmap=cmap,
        alpha=0.5,
    )
    plt.colorbar()
    plt.savefig("{}scatter_{}.png".format(path, label), dpi=400)
    plt.close()
