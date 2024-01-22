from ssumo.data.dataset import fwd_kin_cont6d_torch

from torch.utils.data import DataLoader
from dappy import read
import torch
from dappy import vis
import ssumo

analysis_key = "gre1_b1_x360"
base_path = "/mnt/ceph/users/jwu10/results/vae/heading/"
config = read.config(base_path + analysis_key + "/model_config.yaml")
config["model"]["start_epoch"] = 70
config["model"]["load_model"] = config["out_path"]
config["data"]["stride"] = 10

connectivity = read.connectivity_config(config["data"]["skeleton_path"])

### Load Datasets
train_dataset = ssumo.data.get_mouse(
    data_config=config["data"],
    window=config["model"]["window"],
    train=True,
    data_keys=["x6d", "root", "offsets", "raw_pose"] + config["disentangle"]["features"],
)

test_dataset = ssumo.data.get_mouse(
    data_config=config["data"],
    window=config["model"]["window"],
    train=False,
    data_keys=["x6d", "root", "offsets", "raw_pose"] + config["disentangle"]["features"],
)

batch_size = 10
train_loader = DataLoader(dataset=train_dataset, batch_size=batch_size, shuffle=True)
test_loader = DataLoader(dataset=test_dataset, batch_size=batch_size, shuffle=True)

vae, device = ssumo.model.get(
    model_config=config["model"],
    disentangle_config=config["disentangle"],
    n_keypts = train_dataset.n_keypts,
    direction_process=config["data"]["direction_process"],
    arena_size = train_dataset.arena_size,
    kinematic_tree = train_dataset.kinematic_tree,
    verbose=1,
)
kinematic_tree = train_dataset.kinematic_tree
n_keypts = train_dataset.n_keypts

# vae = utils.init_model(config, n_keypts, config["invariant"]).cuda()
# arena_size = None if config["arena_size"] is None else train_dataset.arena_size.cuda()


def visualize_reconstruction(loader, label):

    vae.eval()
    with torch.no_grad():
        # Let's see how reconstruction looks on train data
        data = next(iter(loader))
        data = {k: v.to(device) for k, v in data.items()}
        data_o = ssumo.train.predict_batch(vae, data, disentangle_keys=config["disentangle"]["features"])

        # x6d = batch["local_6d"].cuda()
        # offsets = batch["offsets"].cuda().view(-1, n_keypts, 3)

        # invariant = batch[config["invariant"]].cuda() if config["invariant"] else None

        # if config["arena_size"] is not None:
        #     root = batch["root"].cuda()
        #     x_i = torch.cat((x6d.view(x6d.shape[:2] + (-1,)), root), axis=-1)
        #     x_o = vae(x_i, invariant=invariant)[0]

        #     x6d_o = x_o[..., :-3]
        #     root_o = inv_normalize_root(x_o[..., -3:].reshape(-1, 3), arena_size)
        #     root = inv_normalize_root(root.reshape(-1, 3), arena_size)
        # else:
        #     root = torch.zeros((len(x6d) * config["window"], 3)).cuda()
        #     root_o = torch.zeros((len(x6d) * config["window"], 3)).cuda()
        #     x6d_o = vae(x6d, invariant=invariant)[0]

        # raw_pose = batch["raw_pose"].cuda().reshape(-1, n_keypts, 3)

        # if config["face_xpos"]:
        #     raw_pose = preprocess.rotate_spine(
        #         preprocess.center_spine(raw_pose.cpu().detach().numpy(), keypt_idx=0),
        #         keypt_idx=[0, 1],
        #         lock_to_x=False,
        #     )
        #     raw_pose = torch.tensor(raw_pose, dtype=x6d.dtype).cuda()
        # import pdb; pdb.set_trace()
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
            data["offsets"].view(-1, n_keypts,3),
            data_o["root"].reshape(-1, 3),
            do_root_R=True,
        )

        pose_array = torch.cat([data["raw_pose"].reshape(-1, n_keypts, 3), pose, pose_hat], axis=0)

        vis.pose.grid3D(
            pose_array.cpu().detach().numpy(),
            connectivity,
            frames=[
                0,
                batch_size * config["model"]["window"],
                2 * batch_size * config["model"]["window"],
            ],
            centered=False,
            subtitles=["Raw", "Raw -> 6D -> Back", "VAE Reconstructed"],
            title=label + " Data",
            fps=45,
            figsize=(36, 12),
            N_FRAMES=batch_size * config["model"]["window"],
            VID_NAME=label + ".mp4",
            SAVE_ROOT=base_path + analysis_key + "/",
        )


visualize_reconstruction(train_loader, "Train")
visualize_reconstruction(test_loader, "Test")
