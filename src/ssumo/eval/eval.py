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
    label,
    minmax=10,
    n_shifts=15,
    grid_vis=True,
    arena_vis=True,
    static_vis=False,
    circle=False,
    save_path="./",
):
    n_keypts = dataset.n_keypts
    window = vae.window

    if circle:
        import scipy.linalg as spl

        linspace = torch.linspace(-torch.pi, torch.pi, n_shifts)[:, None].cuda()
        circ = torch.cat([torch.sin(linspace), torch.cos(linspace)], dim=-1)

        radius = torch.linalg.norm(z[index : index + 1] @ weight.T)

        circ = circ*radius

        z_null_proj = weight.T @ torch.linalg.solve(weight @ weight.T, weight @ z[index: index+1].T)
        circle_z = circ @ weight.cuda()
        circle_z = circle_z/torch.linalg.norm(circle_z,dim=-1)[:,None] * radius

        sample_latent = z[index: index+1].cuda() - z_null_proj.T.cuda() + circle_z

    else:
        graded_z_shift = (
            torch.linspace(-minmax, minmax, n_shifts)[:, None].cuda()
            @ weight.sum(dim=0, keepdim=True).cuda()
        )
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

    PARTS = ["Root", "Head + Spine", "L-Arm", "R-Arm", "L-Leg", "R-Leg"]
    part_lbl = ""
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
                FIG_NAME=dataset.label + "{}_trace_{}.png".format(part_lbl, index),
                SAVE_ROOT=save_path,
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
            title=dataset.label + " Data - {} Traversal".format(label),
            fps=15,
            N_FRAMES=vae.window,
            VID_NAME=dataset.label + "grid{}_mod.mp4".format(index),
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
            VID_NAME=dataset.label + "arena{}_mod.mp4".format(index),
            SAVE_ROOT=save_path,
        )
