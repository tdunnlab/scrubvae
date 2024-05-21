import numpy as np
import matplotlib.pyplot as plt
import ssumo
from dappy import read
import torch
from ssumo.data.dataset import get_angle2D, get_frame_yaw, fwd_kin_cont6d_torch, get_speed_parts
import tqdm
from sklearn.metrics import r2_score
from base_path import RESULTS_PATH, CODE_PATH
import sys
from pathlib import Path
import pickle
palette = ssumo.plot.constants.PALETTE_2
# models = {
#     # "Vanilla VAE": ["vanilla/64", 300],
#     # "Beta VAE": [],
#     "Condtional VAE": ["mals_64/cvae_64", 280],
#     "Gradient Reversal": ["gr_64/50", 290],
#     "Recursive Least Squares": ["mals_64/mals_p1_50", 265],
#     "Mutual Information": ["mi_64/bw75_500", 220],
# }

experiment_folder = sys.argv[1]
disentanglement_key = sys.argv[2] # heading or avg_speed_3d
task_id = sys.argv[3] if len(sys.argv) > 3 else ""

if experiment_folder.endswith(".yaml"):
    config_path = CODE_PATH + "/configs/" + experiment_folder
    config = read.config(config_path)
    analysis_keys = config["PATHS"]
    out_path = config["OUT"]
elif Path(RESULTS_PATH + experiment_folder).is_dir():
    analysis_keys = [experiment_folder]
    out_path = RESULTS_PATH + experiment_folder

if task_id.isdigit():
    analysis_keys = [analysis_keys[int(task_id)]]

metrics_full = {}
for an_key in analysis_keys:
    path = "{}/{}/".format(RESULTS_PATH, an_key)
    config = read.config(path + "/model_config.yaml")
    pickle_path = "{}/genres2_{}.p".format(config["out_path"], disentanglement_key)
    print(pickle_path)
    if Path(pickle_path).is_file():
        metrics = pickle.load(open(pickle_path, "rb"))
        epochs_to_test = [
            e for e in ssumo.get.all_saved_epochs(path) if (e not in metrics["epochs"]) and (e>100)
        ]
        metrics["epochs"] = np.concatenate(
            [metrics["epochs"], epochs_to_test]
        ).astype(int)
    else:
        metrics = {"R2": []}
        if disentanglement_key == "heading":
            metrics["JPE"] = []
        metrics["epochs"] = [e for e in ssumo.get.all_saved_epochs(path) if (e>100)]
        epochs_to_test = metrics["epochs"]

    if len(epochs_to_test) > 0:
        loader = ssumo.get.mouse_data(
            data_config=config["data"],
            window=config["model"]["window"],
            train=True,
            data_keys=["x6d", "root", "offsets",
                "avg_speed_3d",
                "heading", "raw_pose",
            ],
            normalize=[],
            shuffle=False,
        )
        kinematic_tree = loader.dataset.kinematic_tree
        n_keypts = loader.dataset.n_keypts
        arena_size = loader.dataset.arena_size
        discrete_classes = loader.dataset.discrete_classes

        torch.manual_seed(0)
        heading1D_rand = (torch.rand(len(loader.dataset)) * 2 - 1)[:, None] * np.pi
        heading2D_rand = torch.tensor(get_angle2D(heading1D_rand))
        heading_true = loader.dataset[:]["heading"]
        heading1D_true = np.arctan2(heading_true[:, 0].numpy(), heading_true[:, 1].numpy())
        raw_pose = loader.dataset[:]["raw_pose"].numpy()
        raw_pose -= raw_pose[:, config["model"]["window"]//2, 0, :][:, None, None, :]

        rot_angle = (-heading1D_rand.ravel().numpy() + heading1D_true)
        rot_mat = np.array(
            [
                [np.cos(rot_angle), -np.sin(rot_angle), np.zeros(len(rot_angle))],
                [np.sin(rot_angle), np.cos(rot_angle), np.zeros(len(rot_angle))],
                [np.zeros(len(rot_angle)), np.zeros(len(rot_angle)), np.ones(len(rot_angle))],
            ]
        ).repeat(loader.dataset.n_keypts*config["model"]["window"], axis=2)

        pose_rot = np.einsum("jki,ik->ij", rot_mat, raw_pose.reshape(-1, 3)).reshape(
                raw_pose.shape
            )

        spd_mean = loader.dataset[:]["avg_speed_3d"].mean(dim=0)
        spd_std = loader.dataset[:]["avg_speed_3d"].std(dim=0)
        spd_true = (loader.dataset[:]["avg_speed_3d"] - spd_mean)/spd_std
        avg_speed_3d_rand = spd_true[torch.randperm(len(loader.dataset))]
        # avg_speed_3d_rand = torch.clamp(torch.randn_like(spd_true),-2, 2)

    for _, epoch in enumerate(epochs_to_test):
        model = ssumo.get.model(
            model_config=config["model"],
            load_model=config["out_path"],
            epoch=epoch,
            disentangle_config=config["disentangle"],
            n_keypts=n_keypts,
            direction_process=config["data"]["direction_process"],
            loss_config=config["loss"],
            arena_size=arena_size,
            kinematic_tree=kinematic_tree,
            bound=config["data"]["normalize"] is not None,
            discrete_classes=discrete_classes,
            verbose=-1,
        )

        z = ssumo.get.latents(config, model, epoch, loader, "cuda", "Train")
        # torch.tensor(
        #     np.load(path + "latents/Train_{}.npy".format(epoch))
        # )

        if disentanglement_key == "heading":
            z = torch.cat(
                [z, spd_true, heading2D_rand],
                dim=-1,
            )
        elif disentanglement_key == "avg_speed_3d":
            if "heading" in model.conditional_keys:
                z = torch.cat(
                    [z, avg_speed_3d_rand, heading_true],
                    dim=-1,
                )
            else:
                z = torch.cat(
                    [z, avg_speed_3d_rand],
                    dim=-1,
                )

        loader.dataset.data["z"] = z
        pose = []
        for batch_idx, data in enumerate(tqdm.tqdm(loader)):
            x_hat = model.decoder(data["z"].cuda()).moveaxis(-1,1)
            x6d = x_hat[..., :-3].reshape(data["z"].shape[0], model.window, -1, 6)
            root = model.inv_normalize_root(x_hat[..., -3:]).reshape(
                data["z"].shape[0], model.window, 3
            )
            offsets = data["offsets"].cuda()

            pose_batch = fwd_kin_cont6d_torch(
                    x6d.reshape((-1, n_keypts, 6)),
                    kinematic_tree, 
                    offsets.reshape((-1,) + offsets.shape[-2:]),
                    root_pos=root.reshape((-1, 3)),
                    do_root_R=True,
                    eps=1e-8,
                ).reshape((-1, model.window, n_keypts, 3))
            
            pose += [pose_batch.detach().cpu()]
            
        pose = torch.cat(pose, axis=0).numpy()
        
        if disentanglement_key == "heading":
            heading1D_out = get_frame_yaw(pose[:, model.window//2, ...], 0, 1)[..., None]
            pred_out = get_angle2D(heading1D_out)
            metrics["R2"] += [r2_score(heading2D_rand, pred_out)]
            metrics["JPE"] += [((pose_rot - pose)**2).sum()/pose_rot.shape[0]]

        elif disentanglement_key == "avg_speed_3d":
            # speed = np.diff(pose, n=1, axis=1)
            # speed = np.sqrt((speed**2).sum(axis=-1)).mean(axis=-1, keepdims=True)
            # pred_out = np.diff(pose, axis=1)
            speed = get_speed_parts(
                pose=pose.reshape((-1, n_keypts, 3)),
                parts=[
                    [0, 1, 2, 3, 4, 5],  # spine and head
                    [1, 6, 7, 8, 9, 10, 11],  # arms from front spine
                    [5, 12, 13, 14, 15, 16, 17],  # left legs from back spine
                ],
            )
            speed_3d_out = np.concatenate(
                [speed[:, :2], speed[:, 2:].mean(axis=-1, keepdims=True)], axis=-1
            )
            # speed_3d_out = np.concatenate([np.zeros((1,3)), speed_3d_out])
            pred_out = speed_3d_out.reshape((-1, model.window, 3))[:, 1:, ...].mean(axis=1)

            metrics["R2"] += [r2_score(avg_speed_3d_rand, (pred_out-spd_mean.numpy())/spd_std.numpy())]
    
        print(metrics["R2"])
        print(metrics["JPE"])
        pickle.dump(
            metrics,
            open(pickle_path, "wb"),
        )

    # if disentanglement_key == "heading":
    #     import pdb; pdb.set_trace()
    #     for i in range(len(metrics["JPE"])):
    #         if isinstance(metrics["JPE"][i], np.ndarray):
    #             metrics["JPE"][i] = metrics["JPE"][i].sum()

    #     import pdb; pdb.set_trace()
    #     pickle.dump(
    #         metrics,
    #         open(pickle_path, "wb"),
    #     )

    metrics_full[an_key] = metrics

if task_id == "":
    ## Plot R^2
    f = plt.figure(figsize=(15, 10))
    plt.title("Generator Restrictiveness - {}".format(disentanglement_key))
    for path_i, p in enumerate(analysis_keys):
        plt.plot(
            metrics_full[p]["epochs"],
            metrics_full[p]["R2"],
            label=p,
            color=palette[path_i],
            alpha=0.5,
        )

    plt.ylabel("R2")
    plt.legend()
    plt.xlabel("Epoch")

    f.tight_layout()
    plt.savefig("{}/genres_{}.png".format(out_path, disentanglement_key))
    plt.close()

    if disentanglement_key == "heading":
        f = plt.figure(figsize=(15, 10))
        plt.title("Generator Restrictiveness - {}".format(disentanglement_key))
        for path_i, p in enumerate(analysis_keys):

            plt.plot(
                metrics_full[p]["epochs"],
                metrics_full[p]["JPE"],
                label=p,
                color=palette[path_i],
                alpha=0.5,
            )

        plt.ylabel("JPE")
        plt.legend()
        plt.xlabel("Epoch")

        f.tight_layout()
        plt.savefig("{}/genres_jpe_{}.png".format(out_path, disentanglement_key))
        plt.close()