from ssumo.data.dataset import fwd_kin_cont6d_torch

from torch.utils.data import DataLoader
from dappy import read
import torch
from dappy import vis
import ssumo
from base_path import RESULTS_PATH
import sys
import numpy as np


def visualize_2D_reconstruction(model, loader, label, connectivity):
    n_keypts = loader.dataset.n_keypts
    kinematic_tree = loader.dataset.kinematic_tree
    # model.eval()
    with torch.no_grad():
        # Let's see how reconstruction looks on train data
        data = next(iter(loader))
        data = {k: v.to("cuda") for k, v in data.items()}
        # data_o = ssumo.train.predict_batch(
        #     model, data, disentangle_keys=config["disentangle"]["features"]
        # )

        # pose_hat = fwd_kin_cont6d_torch(
        #     data_o["x6d"].reshape(-1, n_keypts, 6),
        #     kinematic_tree,
        #     data["offsets"].view(-1, n_keypts, 3),
        #     data_o["root"].reshape(-1, 3),
        #     do_root_R=True,
        # )
        # import pdb

        # pdb.set_trace()
        data["raw_pose"] = data["raw_pose"].reshape(-1, n_keypts, 2)
        rotv = data["raw_pose"][:, 1] - data["raw_pose"][:, 0]
        rotv = torch.nn.functional.normalize(rotv)
        rotv = rotv.cpu().numpy()
        rotm = torch.tensor([[-rotv[:, 0], rotv[:, 1]], [-rotv[:, 1], -rotv[:, 0]]])
        # rotm = torch.tensor([[rotv[:, 0], rotv[:, 1]], [-rotv[:, 1], rotv[:, 0]]])
        rotm = rotm.swapaxes(0, 2).swapaxes(1, 2)
        data["raw_pose"] = data["raw_pose"] @ rotm.to("cuda")
        pose_array = torch.cat(
            [
                data["raw_pose"],
                # data["raw_pose"].reshape(-1, n_keypts, 2),
                data["target_pose"].reshape(-1, n_keypts, 2),
                # pose_hat,
            ],
            axis=0,
        )
        vis.pose.grid2D(
            pose_array.cpu().detach().numpy(),
            connectivity,
            frames=[
                0,
                config["data"]["batch_size"] * config["model"]["window"],
                # 2 * config["data"]["batch_size"] * config["model"]["window"],
            ],
            centered=False,
            # subtitles=["Raw", "Target"],  # "Reconstructed"],
            subtitles=["Spine Locked", "Upright Camera"],
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
    # loader, model = ssumo.get.data_and_model(
    #     config,
    #     load_model=config["out_path"],
    #     epoch=sys.argv[2],
    #     dataset_label="Train",
    #     data_keys=["x6d", "root", "offsets", "raw_pose", "target_pose"]
    #     + config["disentangle"]["features"],
    #     shuffle=True,
    #     verbose=0,
    # )
    loader = ssumo.get.mouse_data(
        data_config=config["data"],
        window=config["model"]["window"],
        train=dataset_label == "Train",
        data_keys=["x6d", "root", "offsets", "target_pose", "raw_pose"],
        shuffle=True,
        normalize=config["disentangle"]["features"],
        is_2D=True,
    )
    model = 1

    visualize_2D_reconstruction(model, loader, dataset_label, connectivity)
