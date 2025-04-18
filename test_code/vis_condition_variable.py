from ssumo.data.dataset import fwd_kin_cont6d_torch
from ssumo.get.data import get_projected_2D_kinematics

from torch.utils.data import DataLoader
from neuroposelib import read
import torch
from neuroposelib import vis
import ssumo

# from base_path import RESULTS_PATH
RESULTS_PATH = "/mnt/ceph/users/hkoneru/results/vae/"
import sys
import random
import numpy as np
import scipy.linalg as spl


def visualize_conditional_variable_2D(model, loader, label, connectivity, config):
    n_keypts = loader.dataset.n_keypts
    kinematic_tree = loader.dataset.kinematic_tree
    model.eval()
    with torch.no_grad():
        # Let's see how reconstruction looks on train data
        data = next(iter(loader))
        data = {k: v.to("cuda") for k, v in data.items()}
        angles = np.linspace(0, np.pi, config["model"]["window"])
        # angles = np.full(config["model"]["window"], np.pi / 4)
        axis = np.swapaxes(
            np.array([np.zeros_like(angles), -np.sin(angles), -np.cos(angles)]), 0, 1
        )
        axis = np.tile(axis, (config["data"]["batch_size"], 1))
        skeleton_config = read.config(config["data"]["skeleton_path"])
        data = get_projected_2D_kinematics(
            data,
            # torch.tensor([0, -np.sqrt(2) / 2, -np.sqrt(2) / 2])
            torch.tensor([0, -1, 0]).type(torch.FloatTensor).to("cuda"),
            # [0, -1, 0], # these give terrible results
            # [0, 0, -1],
            skeleton_config,
        )
        # data["view_axis"] = (
        #     torch.tensor(axis)[None, :]
        #     .repeat((len(data["3D_pose"]), 1))
        #     .type(torch.FloatTensor)
        # )
        data = {
            k: v.repeat_interleave(config["model"]["window"], dim=0)
            for k, v in data.items()
        }
        data["view_axis"] = torch.from_numpy(axis).type(torch.FloatTensor)
        data = {k: v.to("cuda") for k, v in data.items()}
        data_o = ssumo.train.predict_batch(
            model, data, disentangle_keys=config["disentangle"]["features"]
        )
        x_hat = data_o["x6d"]
        local_ang = x_hat.reshape((-1,) + x_hat.shape[-2:])
        reshaped_x6d = torch.concatenate(
            [local_ang[..., :], torch.zeros_like(local_ang[..., [0]])], axis=-1
        )
        reshaped_x6d = torch.concatenate(
            [reshaped_x6d[..., [1, 0, 2]], reshaped_x6d[..., :]], axis=-1
        )
        reshaped_x6d[..., 3] *= -1

        pose_hat = fwd_kin_cont6d_torch(
            reshaped_x6d,
            kinematic_tree,
            data["offsets"].view(-1, n_keypts, 3),
            torch.zeros(reshaped_x6d.shape[:-2] + (3,)),
            do_root_R=True,
        ).reshape(-1, n_keypts, 3)
        pose_hat = pose_hat[..., :2]
        pose_hat = torch.reshape(  # pose_hat should be 5 videos x 51 axes x 51 frames
            pose_hat,
            (
                config["data"]["batch_size"],
                config["model"]["window"],
                config["model"]["window"],
            )
            + pose_hat.shape[1:],
        )
        pose_hat = (
            torch.diagonal(pose_hat, dim1=1, dim2=2)
            .movedim(3, 1)
            .reshape((-1,) + pose_hat.shape[3:])
        )

        true_rotated = data["raw_pose"][0 :: config["model"]["window"]].reshape(
            -1, n_keypts, 3
        )
        for ind, ax in enumerate(axis):
            uperp = (
                torch.from_numpy(spl.null_space(ax[None, :]))
                .type(torch.FloatTensor)
                .to("cuda")
            )
            if uperp[2][0] == 0:
                proj_x = uperp.T[0]
                proj_y = uperp.T[1]
            else:
                coeff = -uperp[2][1] / uperp[2][0]
                proj_x = uperp.T[0] * coeff + uperp.T[1]
                proj_y = torch.cross(
                    torch.tensor(ax).type(torch.FloatTensor).to("cuda"), proj_x
                )
            proj_x /= torch.norm(proj_x)
            proj_y /= torch.norm(proj_y)
            if proj_y[2] < 0:
                proj_y *= -1
            if np.linalg.norm(torch.cross(proj_x, proj_y).cpu().numpy() - ax) > 0.1:
                proj_x *= -1
            true_rotated[ind, ..., :2] = (
                true_rotated[ind, ...]
                @ torch.cat([proj_x[None, ...], proj_y[None, ...]], axis=0).T
            )
        true_rotated = true_rotated[..., :2]

        pose_array = torch.cat(
            [
                data["projected_pose"][0 :: config["model"]["window"]].reshape(
                    -1, n_keypts, 2
                ),
                true_rotated,
                pose_hat,
            ],
            axis=0,
        )
        vis.pose.grid2D(
            # pose_hat.cpu().detach().numpy(),
            pose_array.cpu().detach().numpy(),
            connectivity,
            frames=[
                0,
                config["data"]["batch_size"] * config["model"]["window"],
                2 * config["data"]["batch_size"] * config["model"]["window"],
            ],
            centered=False,
            subtitles=[
                "2D input",
                "Rotating input pose",
                "Rotating Reconstruction",
            ],
            title=label + " Data",
            fps=30,
            figsize=(24, 8),
            N_FRAMES=config["data"]["batch_size"] * config["model"]["window"],
            VID_NAME=label + "_cond.mp4",
            SAVE_ROOT=config["out_path"],
        )


analysis_key = sys.argv[1]
config = read.config(RESULTS_PATH + analysis_key + "/model_config.yaml")
config["data"]["stride"] = 10
config["data"]["batch_size"] = 5
connectivity = read.connectivity_config(config["data"]["skeleton_path"])
dataset_list = ["Train"]  # "Test"
for dataset_label in dataset_list:
    loader, model = ssumo.get.data_and_model(
        config,
        load_model=config["out_path"],
        epoch=sys.argv[2],
        dataset_label="Train",
        data_keys=[
            "x6d",
            "offsets",
            "raw_pose",
            "target_pose",
            "projected_pose",
        ]
        + config["disentangle"]["features"],
        shuffle=True,
        verbose=0,
    )

    visualize_conditional_variable_2D(
        model, loader, dataset_label, connectivity, config
    )
