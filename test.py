from ssumo.data.dataset import fwd_kin_cont6d_torch

from torch.utils.data import DataLoader
from dappy import read
import torch
from dappy import vis
import ssumo
from base_path import RESULTS_PATH
import sys

analysis_key = sys.argv[1]
config = read.config(RESULTS_PATH + analysis_key + "/model_config.yaml")
config["model"]["start_epoch"] = sys.argv[2]
config["model"]["load_model"] = config["out_path"]
config["data"]["stride"] = 10
config["data"]["batch_size"] = 10

connectivity = read.connectivity_config(config["data"]["skeleton_path"])

### Load Datasets
train_dataset, train_loader = ssumo.data.get_mouse(
    data_config=config["data"],
    window=config["model"]["window"],
    train=True,
    data_keys=["x6d", "root", "offsets", "raw_pose", "target_pose"],
    shuffle=True,
)

test_dataset, test_loader = ssumo.data.get_mouse(
    data_config=config["data"],
    window=config["model"]["window"],
    train=False,
    data_keys=["x6d", "root", "offsets", "raw_pose", "target_pose"],
    shuffle=True,
)

vae, device = ssumo.model.get(
    model_config=config["model"],
    disentangle_config=config["disentangle"],
    n_keypts=train_dataset.n_keypts,
    direction_process=config["data"]["direction_process"],
    arena_size=train_dataset.arena_size,
    kinematic_tree=train_dataset.kinematic_tree,
    verbose=1,
)
kinematic_tree = train_dataset.kinematic_tree
n_keypts = train_dataset.n_keypts

def visualize_reconstruction(loader, label):
    vae.eval()
    with torch.no_grad():
        # Let's see how reconstruction looks on train data
        data = next(iter(loader))
        data = {k: v.to(device) for k, v in data.items()}
        data_o = ssumo.train.predict_batch(
            vae, data, disentangle_keys=config["disentangle"]["features"]
        )

        pose = fwd_kin_cont6d_torch(
            data["x6d"].reshape(-1, n_keypts, 6),
            kinematic_tree,
            data["offsets"].view(-1, n_keypts, 3),
            data["root"].reshape(-1, 3),
            do_root_R=True,
        )

        pose_hat = fwd_kin_cont6d_torch(
            data_o["x6d"].reshape(-1, n_keypts, 6),
            kinematic_tree,
            data["offsets"].view(-1, n_keypts, 3),
            data_o["root"].reshape(-1, 3),
            do_root_R=True,
        )

        pose_array = torch.cat(
            [data["target_pose"].reshape(-1, n_keypts, 3), pose, pose_hat], axis=0
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
            subtitles=["Raw", "Raw -> 6D -> Back", "VAE Reconstructed"],
            title=label + " Data",
            fps=45,
            figsize=(36, 12),
            N_FRAMES=config["data"]["batch_size"] * config["model"]["window"],
            VID_NAME=label + ".mp4",
            SAVE_ROOT=config["out_path"],
        )


visualize_reconstruction(train_loader, "Train")
visualize_reconstruction(test_loader, "Test")
