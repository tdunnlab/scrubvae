from ssumo.data.dataset import fwd_kin_cont6d_torch
from ssumo.get.data import get_projected_2D_kinematics, get_random_axis
from torch.utils.data import DataLoader
from dappy import read
import torch
from dappy import vis
import ssumo
from base_path import RESULTS_PATH
import sys
from math import pi


def visualize_2D_reconstruction(model, loader, label, connectivity, config):
    n_keypts = loader.dataset.n_keypts
    kinematic_tree = loader.dataset.kinematic_tree
    model.eval()
    with torch.no_grad():
        # Let's see how reconstruction looks on train data
        data = next(iter(loader))
        skeleton_config = read.config(config["data"]["skeleton_path"])
        #### Pick projection axes
        # axis = get_random_axis(1).to("cuda") # single random axis for all
        # data["view_axis"] = axis[None, :].repeat((len(data["raw_pose"]), 1))
        axis = get_random_axis(len(data["raw_pose"])).to(
            "cuda"
        )  # different random axis each
        # axis[0] = torch.tensor([0, 0, -1]).to("cuda")
        data["view_axis"] = axis

        data = {k: v.to("cuda") for k, v in data.items()}
        data = get_projected_2D_kinematics(
            data,
            axis,
            skeleton_config,
        )
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

        pose_array = torch.cat(
            [
                data["projected_pose"].reshape(-1, n_keypts, 2),
                data["target_pose"].reshape(-1, n_keypts, 2),
                pose_hat,
            ],
            axis=0,
        )
        vis.pose.grid2D(
            pose_array.cpu().detach().numpy(),
            connectivity,
            frames=[
                0,
                config["data"]["batch_size"] * config["model"]["window"],
                2 * config["data"]["batch_size"] * config["model"]["window"],
            ],
            centered=False,
            subtitles=["Raw", "Target", "Reconstructed"],
            title=label + " Data",
            fps=45,
            figsize=(36, 12),
            N_FRAMES=config["data"]["batch_size"] * config["model"]["window"],
            VID_NAME=label + ".mp4",
            SAVE_ROOT=config["out_path"],
        )


def visualize_reconstruction(model, loader, label, connectivity):
    n_keypts = loader.dataset.n_keypts
    kinematic_tree = loader.dataset.kinematic_tree
    model.eval()
    with torch.no_grad():
        # Let's see how reconstruction looks on train data
        data = next(iter(loader))
        data = {k: v.to("cuda") for k, v in data.items()}
        data_o = ssumo.train.predict_batch(
            model, data, disentangle_keys=config["disentangle"]["features"]
        )

        pose_hat = fwd_kin_cont6d_torch(
            data_o["x6d"].reshape(-1, n_keypts, 6),
            kinematic_tree,
            data["offsets"].view(-1, n_keypts, 3),
            data_o["root"].reshape(-1, 3),
            do_root_R=True,
        )

        pose_array = torch.cat(
            [
                data["raw_pose"].reshape(-1, n_keypts, 3),
                data["target_pose"].reshape(-1, n_keypts, 3),
                pose_hat,
            ],
            axis=0,
        )

        vis.pose.grid3D(
            pose_array.cpu().detach().numpy(),
            connectivity,
            frames=[
                0,
                config["data"]["batch_size"] * config["model"]["window"],
                2 * config["data"]["batch_size"] * config["model"]["window"],
            ],
            centered=False,
            subtitles=["Raw", "Target", "Reconstructed"],
            title=label + " Data",
            fps=45,
            figsize=(36, 12),
            N_FRAMES=config["data"]["batch_size"] * config["model"]["window"],
            VID_NAME=label + ".mp4",
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
            "root",
            "offsets",
            "raw_pose",
            "target_pose",
            "projected_pose",
        ]
        + config["disentangle"]["features"],
        shuffle=True,
        verbose=0,
    )

    # visualize_reconstruction(model, loader, dataset_label, connectivity)
    visualize_2D_reconstruction(model, loader, dataset_label, connectivity, config)
