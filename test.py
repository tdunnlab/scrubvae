from data.dataset import (
    MouseDataset,
    fwd_kin_cont6d_torch,
    inv_normalize_root,
)
from torch.utils.data import DataLoader
from dappy import read, preprocess
import torch
from data.rotation_conversion import rotation_6d_to_matrix
from dappy import vis
import utils

path = "rc_w51_midfwd_full"
base_path = "/mnt/home/jwu10/working/behavior_vae/"
config = utils.read_config(base_path + "/results/" + path + "/model_config.yaml")
config["load_epoch"] = 400
config["load_model"] = config["out_path"]

connectivity = read.connectivity_config(base_path + "/configs/mouse_skeleton.yaml")

if config["arena_size"] is not None:
    train_dataset = MouseMidCentered(
        data_path=config["data_path"],
        skeleton_path=base_path + "/configs/mouse_skeleton.yaml",
        train=True,
        window=config["window"],
        stride=4,
        face_xpos=config["face_xpos"],
        arena_size=config["arena_size"],
        invariant=config["invariant"],
        get_raw_pose=True,
    )

    test_dataset = MouseMidCentered(
        data_path=config["data_path"],
        skeleton_path=base_path + "/configs/mouse_skeleton.yaml",
        train=True,
        window=config["window"],
        stride=4,
        face_xpos=config["face_xpos"],
        arena_size=config["arena_size"],
        invariant=config["invariant"],
        get_raw_pose=True,
    )
else:
    train_dataset = MouseDataset(
        data_path=config["data_path"],
        skeleton_path=base_path + "/configs/mouse_skeleton.yaml",
        train=True,
        window=config["window"],
        stride=4,
        face_xpos=config["face_xpos"],
        arena_size=config["arena_size"],
        invariant=config["invariant"],
        get_raw_pose=True,
    )

    # Load in test dataset
    test_dataset = MouseDataset(
        data_path=config["data_path"],
        skeleton_path=base_path + "/configs/mouse_skeleton.yaml",
        train=False,
        window=config["window"],
        stride=4,
        face_xpos=config["face_xpos"],
        arena_size=config["arena_size"],
        invariant=config["invariant"],
        get_raw_pose=True,
    )

batch_size = 10
train_loader = DataLoader(dataset=train_dataset, batch_size=batch_size, shuffle=True)
test_loader = DataLoader(dataset=test_dataset, batch_size=batch_size, shuffle=True)
kinematic_tree = train_dataset.kinematic_tree
n_keypts = train_dataset.n_keypts

vae = utils.init_model(config, n_keypts, config["invariant"]).cuda()
arena_size = None if config["arena_size"] is None else train_dataset.arena_size.cuda()


def visualize_reconstruction(loader, label):
    vae.eval()
    with torch.no_grad():
        # Let's see how reconstruction looks on train data
        batch = next(iter(loader))
        x6d = batch["local_6d"].cuda()
        offsets = batch["offsets"].cuda().view(-1, n_keypts, 3)

        invariant = batch[config["invariant"]].cuda() if config["invariant"] else None

        if config["arena_size"] is not None:
            root = batch["root"].cuda()
            x_i = torch.cat((x6d.view(x6d.shape[:2] + (-1,)), root), axis=-1)
            x_o = vae(x_i, invariant=invariant)[0]

            x6d_o = x_o[..., :-3]
            root_o = inv_normalize_root(x_o[..., -3:].reshape(-1, 3), arena_size)
            root = inv_normalize_root(root.reshape(-1, 3), arena_size)
        else:
            root = torch.zeros((len(x6d) * config["window"], 3)).cuda()
            root_o = torch.zeros((len(x6d) * config["window"], 3)).cuda()
            x6d_o = vae(x6d, invariant=invariant)[0]

        raw_pose = batch["raw_pose"].cuda().reshape(-1, n_keypts, 3)

        if config["face_xpos"]:
            raw_pose = preprocess.rotate_spine(
                preprocess.center_spine(raw_pose.cpu().detach().numpy(), keypt_idx=0),
                keypt_idx=[0, 1],
                lock_to_x=False,
            )
            raw_pose = torch.tensor(raw_pose, dtype=x6d.dtype).cuda()

        pose = fwd_kin_cont6d_torch(
            x6d.reshape(-1, n_keypts, 6),
            kinematic_tree,
            offsets,
            root,
            do_root_R=True,
        )

        pose_hat = fwd_kin_cont6d_torch(
            x6d_o.reshape(-1, n_keypts, 6),
            kinematic_tree,
            offsets,
            root_o,
            do_root_R=True,
        )

        pose_array = torch.cat([raw_pose, pose, pose_hat], axis=0)

        vis.pose.grid3D(
            pose_array.cpu().detach().numpy(),
            connectivity,
            frames=[
                0,
                batch_size * config["window"],
                2 * batch_size * config["window"],
            ],
            centered=False,
            subtitles=["Raw", "Raw -> 6D -> Back", "VAE Reconstructed"],
            title=label + " Data",
            fps=45,
            figsize=(36, 12),
            N_FRAMES=batch_size * config["window"],
            VID_NAME=label + ".mp4",
            SAVE_ROOT=base_path + "/results/" + path + "/",
        )


visualize_reconstruction(train_loader, "Train")
visualize_reconstruction(test_loader, "Test")
