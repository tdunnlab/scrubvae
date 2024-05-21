import numpy as np
import matplotlib.pyplot as plt
import ssumo
from dappy import read, preprocess
import torch
from ssumo.data.dataset import (
    get_angle2D,
    get_frame_yaw,
    fwd_kin_cont6d_torch,
    get_speed_parts,
)
import tqdm
from sklearn.metrics import r2_score
import sys
from pathlib import Path
import pickle

plt.rcParams["font.family"] = "sans-serif"
plt.rcParams["font.sans-serif"] = "Arial"
plt.rcParams["font.size"] = "10"

palette = ssumo.plot.constants.PALETTE_2

CODE_PATH = "/mnt/home/jwu10/working/ssumo/"
RESULTS_PATH = "/mnt/ceph/users/jwu10/results/vae/"

titles = {
    "heading": "Heading Direction",
    "avg_speed_3d": "Average Speed",
    "ids": "Animal ID",
}

downsample = 32
f = plt.figure(figsize=(12, 8))
subf = f.subfigures(2, 1)
for var_ind, var_key in enumerate(["avg_speed_3d", "heading"]):
    print(var_key)
    models = read.config(CODE_PATH + "configs/exp_finals.yaml")[var_key]
    models = {m[0]: [m[1], m[2]] for m in models if (m[0] != "Vanilla VAE")}

    torch.manual_seed(0)
    config = read.config(
        RESULTS_PATH + models["Mutual Information"][0] + "/model_config.yaml"
    )
    # config["data"]["stride"] = downsample
    if var_key == "heading":
        loader = ssumo.get.mouse_data(
            data_config=config["data"],
            window=config["model"]["window"],
            train=True,
            data_keys=[
                "x6d",
                "root",
                "offsets",
                "avg_speed_3d",
                "heading",
                "raw_pose",
            ],
            normalize=["avg_speed_3d"],
            shuffle=False,
        )

        heading1D_rand = (
            torch.rand(len(loader.dataset[::downsample]["root"])) * 2 - 1
        )[:, None] * np.pi
        heading2D_rand = torch.tensor(get_angle2D(heading1D_rand))
        heading_true = loader.dataset[::downsample]["heading"]
        heading1D_true = np.arctan2(
            heading_true[:, 0].numpy(), heading_true[:, 1].numpy()
        )
        # raw_pose = loader.dataset[::downsample]["raw_pose"].numpy()
        # raw_pose -= raw_pose[:, config["model"]["window"] // 2, 0, :][:, None, None, :]

        # rot_angle = (-heading1D_rand.ravel().numpy() + heading1D_true)
        # rot_mat = np.array(
        #     [
        #         [np.cos(rot_angle), -np.sin(rot_angle), np.zeros(len(rot_angle))],
        #         [np.sin(rot_angle), np.cos(rot_angle), np.zeros(len(rot_angle))],
        #         [np.zeros(len(rot_angle)), np.zeros(len(rot_angle)), np.ones(len(rot_angle))],
        #     ]
        # ).repeat(loader.dataset.n_keypts*config["model"]["window"], axis=2)

        # pose_rot = np.einsum("jki,ik->ij", rot_mat, raw_pose.reshape(-1, 3)).reshape(
        #         raw_pose.shape
        #     )
        raw_pose = loader.dataset.data["raw_pose"].numpy()
        pose_rot = preprocess.rotate_spine(
            preprocess.center_spine(raw_pose, keypt_idx=0), keypt_idx=[0, 1]
        )
        pose_rot = pose_rot[loader.dataset.window_inds[::downsample], ...]
        spd_true = loader.dataset[::downsample]["avg_speed_3d"]

    elif var_key == "avg_speed_3d":
        loader = ssumo.get.mouse_data(
            data_config=config["data"],
            window=config["model"]["window"],
            train=True,
            data_keys=["avg_speed_3d", "offsets"],
            normalize=[],
            shuffle=False,
        )

        spd_mean = loader.dataset[:]["avg_speed_3d"].mean(dim=0)
        spd_std = loader.dataset[:]["avg_speed_3d"].std(dim=0)
        print("STD MEAN")
        print(spd_std)
        print(spd_mean)
        spd_true = (loader.dataset[::downsample]["avg_speed_3d"] - spd_mean) / spd_std
        # avg_speed_3d_rand = spd_true[torch.randperm(len(spd_true))]
        rand_jitter = torch.randn((spd_true.shape[0], 1))*spd_std*1.5+0.5

        avg_speed_3d_rand = spd_true + rand_jitter # [torch.randperm(len(spd_true))]

        # for dim in range(avg_speed_3d_rand.shape[-1]):
        #     import pdb; pdb.set_trace()
        #     avg_speed_3d_rand[:,dim] = torch.clamp(avg_speed_3d_rand[:,dim], min=spd_true.min(dim=0)[0][dim], max=None)
        avg_speed_3d_rand = torch.clamp(avg_speed_3d_rand, min=spd_true.min(dim=0)[0], max=spd_true.max(dim=0)[0])

        kinematic_tree = loader.dataset.kinematic_tree
        n_keypts = loader.dataset.n_keypts
        arena_size = loader.dataset.arena_size
        discrete_classes = loader.dataset.discrete_classes

    loader.dataset.window_inds = loader.dataset.window_inds[::downsample]
    assert len(loader.dataset.window_inds) == len(spd_true)
    gs = subf[var_ind].add_gridspec(1, 4)

    if var_key == "heading":
        # gs2 = subf[var_ind + 1].add_gridspec(1, 4)
        subf[var_ind].suptitle(
            "Generated {} For Random Input {}".format(titles[var_key], titles[var_key]), fontsize=14
        )
        # subf[var_ind + 1].suptitle(
        #     "Generated Joint Position Error For Random {}".format(titles[var_key])
        # )
        # jpe_ax = []
        # jpe_max = 0
        # jpe_min = 100

    elif var_key == "avg_speed_3d":
        subf[var_ind].suptitle(
            "Generated {} For Random Input {}".format(titles[var_key], titles[var_key]), fontsize=14
        )

    for i, model_key in enumerate(models.keys()):
        print(model_key)
        path = "{}{}/".format(RESULTS_PATH, models[model_key][0])
        config = read.config(path + "/model_config.yaml")
        model = ssumo.get.model(
            model_config=config["model"],
            load_model=config["out_path"],
            epoch=models[model_key][1],
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

        z = torch.tensor(
            np.load(path + "latents/Train_{}.npy".format(models[model_key][1]))[
                ::downsample
            ]
        )

        if var_key == "heading":
            z = torch.cat(
                [z, spd_true, heading2D_rand],
                dim=-1,
            )
        elif var_key == "avg_speed_3d":
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
        model.eval()
        for batch_idx, data in enumerate(tqdm.tqdm(loader)):
            x_hat = model.decoder(data["z"].cuda()).moveaxis(-1, 1)
            x6d = x_hat[..., :-3].reshape(data["z"].shape[0], model.window, -1, 6)
            offsets = data["offsets"].cuda()

            if var_key == "heading":
                root = torch.zeros((data["z"].shape[0]* model.window, 3), device=x6d.device)
            elif var_key == "avg_speed_3d":
                root = model.inv_normalize_root(x_hat[..., -3:]).reshape(
                data["z"].shape[0]* model.window, 3
            )
            pose_batch = fwd_kin_cont6d_torch(
                x6d.reshape((-1, n_keypts, 6)),
                kinematic_tree,
                offsets.reshape((-1,) + offsets.shape[-2:]),
                root_pos=root,
                do_root_R=True,
                eps=1e-8,
            ).reshape((-1, model.window, n_keypts, 3))

            pose += [pose_batch.detach().cpu()]

        pose = torch.cat(pose, axis=0).numpy()

        if var_key == "heading":
            heading1D_out = get_frame_yaw(pose[:, model.window // 2, ...], 0, 1)[
                ..., None
            ]
            pred_out = get_angle2D(heading1D_out)
            # r2 = np.mean(np.median(np.abs(heading2D_rand-pred_out)))
            r2 = r2_score(heading2D_rand, pred_out)
            pose_out_rot = preprocess.rotate_spine(
                preprocess.center_spine(pose.reshape((-1, n_keypts, 3)), 0), [0, 1]
            )
            jpe = np.sqrt(((pose_rot - pose_out_rot.reshape(pose_rot.shape)) ** 2).sum(axis=-1)).mean(axis=(-1, -2))
            mean_jpe = jpe.mean()
            expected = heading1D_rand
            pred_out = np.arctan2(pred_out[:, 0], pred_out[:, 1])

        elif var_key == "avg_speed_3d":
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
            pred_out = speed_3d_out.reshape((-1, model.window, 3))[:, 1:, ...].mean(
                axis=1
            )

            r2 = r2_score(
                avg_speed_3d_rand, (pred_out - spd_mean.numpy()) / spd_std.numpy()
            )
            # r2 = np.mean(np.median(np.abs(avg_speed_3d_rand-(pred_out-spd_mean.numpy())/spd_std.numpy())))

            expected = (avg_speed_3d_rand * spd_std + spd_mean).mean(axis=1)
            pred_out = pred_out.mean(axis=-1)

        ax = subf[var_ind].add_subplot(gs[i])
        ax.set_title(model_key + "\n$R^2 = ${:3f}".format(r2))
        y_eq_x = np.linspace(min(expected.min(), pred_out.min()), max(expected.max(), pred_out.max()),1000)
        ax.plot(y_eq_x, y_eq_x, label="$y=x$", c="k", lw=1)
        ax.scatter(expected, pred_out, s=3, marker="o", facecolors="none", edgecolors="C0")
        # xy_min = min(expected.min(), pred_out.min())
        # xy_max = max(expected.max(), pred_out.max())
        # ax.plot(np.linspace(xy_min, xy_max, 1000), np.linspace(xy_min, xy_max, 1000))
        # ax.text(x=(expected.max() + expected.min())/2, y= pred_out.max()+0.1, s="$R^2 = $" + "{:3f}".format(r2))
        # ax.text(x=0.25, y=0.9, s="$R^2 = ${:3f}".format(r2), transform=ax.transAxes)
        # ax.text(x=0.33, y=0.9, s=r"MAD$ = {:3f}$".format(r2), transform=ax.transAxes)
        ax.set_xlabel("Input {}".format(titles[var_key]))
        ax.set_ylabel("Generated {}".format(titles[var_key]))
        ax.legend(loc="lower right" if var_key == "avg_speed_3d" else "lower center")

        # if var_key == "heading":
        #     jpe_max = max(jpe_max, jpe.max())
        #     jpe_min = min(jpe_min, jpe.min())
        #     ax = subf[var_ind + 1].add_subplot(gs2[i])
        #     ax.set_title(model_key)
        #     ax.scatter(expected, jpe, s=3, marker="o", facecolors="none", edgecolors="C0")
        #     ax.hlines(
        #         mean_jpe,
        #         expected.min(),
        #         expected.max(),
        #         linestyles="dashed",
        #         colors="black",
        #         label=r"$\mu = {:3f}$".format(mean_jpe),
        #     )
        #     ax.set_xlabel("Input {}".format(titles[var_key]))
        #     ax.set_ylabel("Joint Position Error")
        #     ax.legend()
        #     jpe_ax += [ax]

    subf[var_ind].subplots_adjust(left=0.05,
                            bottom=0.15, 
                            right=0.98, 
                            top=0.75,
                            wspace=0.3, 
                            hspace=0.1)

    # if var_key == "heading":
    #     for i, ax in enumerate(jpe_ax):
    #         ax.set_ylim(bottom=jpe_min - 0.1, top=jpe_max + 0.1)

# f.tight_layout()
plt.savefig("./results/genres_final.png")
