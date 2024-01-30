import scipy.linalg as spl
from ssumo.data.dataset import fwd_kin_cont6d_torch
import numpy as np
import torch
from ssumo.plot import trace, PLANE
from dappy import visualization as vis

def project_to_null(z, weight):
    print("Finding null space projection of decoder ...")
    u_orth = spl.null_space(weight)
    z_null = z @ u_orth

    return z_null, u_orth


def traverse_latent(
    vae,
    dataset,
    z,
    weight,
    index,
    connectivity,
    minmax=10,
    n_shifts=15,
    grid_vis=True,
    arena_vis=True,
    static_vis=False,
):
    n_keypts = vae.n_keypts
    window = dataset.window
    graded_z_shift = ( torch.linspace(-minmax, minmax, n_shifts)[:,None].cuda() @ weight.sum(dim=0,keepdim=True).cuda() )
    sample_latent = z[index : index + 1].repeat(n_shifts, 1).cuda()
    print("Latent Norm: {}".format(torch.linalg.norm(sample_latent[0])))
    sample_latent += graded_z_shift

    data_o = vae.decode(sample_latent)
    offsets = dataset[index]["offsets"].cuda()
    pose = (
        fwd_kin_cont6d_torch(
            data_o["x6d"].reshape((-1, n_keypts, 6)),
            vae.kinematic_tree,
            offsets.repeat(n_shifts, 1, 1),
            root_pos=data_o["root"].reshape((-1, 3)),
            do_root_R=True,
        )
        .cpu()
        .detach()
        .numpy()
    )

    # part_lbl = ""
    # if config["speed_decoder"] == "part":
    #     part_lbl = PARTS[i]

    if static_vis:
        for vis_plane in ["xz", "xy"]:
            pose_trans = pose.reshape(n_shifts, window, n_keypts, 3)
            pose_trans[..., PLANE[vis_plane[-1]]] += +(
                np.linspace(-20, 20, n_shifts) * n_shifts
            )[:, None, None]
            trace(
                pose_trans.reshape(-1, n_keypts, 3),
                connectivity,
                frames=np.arange(n_shifts) * window,
                n_full_pose=3,
                vis_plane=vis_plane,
                centered=False,
                N_FRAMES=window,
                FIG_NAME=dataset_label + "{}_trace_{}.png".format(part_lbl, index),
                SAVE_ROOT=vis_decode_path,
            )

    subtitles = (sample_latent @ weight.T.cuda()).detach().cpu().numpy()
    if weight.shape[0] != 1:
        subtitles = [
            " ".join(["{:.2f}".format(ss) for ss in s]) for s in subtitles.squeeze()
        ]
    else:
        subtitles = ["{:.2f}".format(s) for s in subtitles.squeeze()]

    if grid_vis:
        vis.pose.grid3D(
            pose,
            connectivity,
            frames=np.arange(n_shifts) * vae.window,
            centered=False,
            subtitles=subtitles,
            title=dataset_label + " Data - {} Traversal".format(k),
            fps=15,
            N_FRAMES=vae.window,
            VID_NAME=dataset_label + "grid{}_mod.mp4".format(index),
            SAVE_ROOT=save_path,
        )

    if arena_vis:
        vis.pose.arena3D(
            pose,
            connectivity,
            frames=np.arange(n_shifts) * vae.window,
            centered=False,
            fps=15,
            N_FRAMES=vae.window,
            VID_NAME=dataset_label + "arena{}_mod.mp4".format(index),
            SAVE_ROOT=save_path,
        )
