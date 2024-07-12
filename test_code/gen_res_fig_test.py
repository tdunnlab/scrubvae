import numpy as np
import matplotlib.pyplot as plt
import ssumo
from neuroposelib import read, preprocess
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
f = plt.figure(figsize=(12, 6))
subf = f.subfigures(2, 1)
for var_ind, var_key in enumerate(["avg_speed_3d","heading"]):
    print(var_key)
    models = read.config(CODE_PATH + "configs/exp_finals.yaml")[var_key]
    models = {m[0]: [m[1], m[2]] for m in models if (m[0] != "VAE")}

    torch.manual_seed(0)
    config = read.config(RESULTS_PATH + models["SC-VAE-MI"][0] + "/model_config.yaml")
    config["data"]["stride"] = downsample
    if var_key == "heading":
        loader = ssumo.get.mouse_data(
            data_config=config["data"],
            window=config["model"]["window"],
            train=False,
            data_keys=[
                "x6d",
                "root",
                "offsets",
                "avg_speed_3d",
                "heading",
                # "raw_pose",
            ],
            normalize=[],
            shuffle=False,
        )

        heading1D_rand = (
            torch.rand(len(loader.dataset.data["root"])) * 2 - 1
        )[:, None] * np.pi
        heading2D_rand = torch.tensor(get_angle2D(heading1D_rand))
        heading_true = loader.dataset.data["heading"]
        # heading2D_rand = heading_true
        heading1D_true = np.arctan2(
            heading_true[:, 0].numpy(), heading_true[:, 1].numpy()
        )
        # heading1D_rand = heading1D_true
        loader.dataset.data["heading"] = heading2D_rand

        spd_std = torch.tensor([0.5616, 0.3610, 0.4162])
        spd_mean = torch.tensor([0.6783, 0.7000, 0.6396])
        spd_true = (loader.dataset.data["avg_speed_3d"] - spd_mean) / spd_std
        loader.dataset.data["avg_speed_3d"] = spd_true

    elif var_key == "avg_speed_3d":
        loader = ssumo.get.mouse_data(
            data_config=config["data"],
            window=config["model"]["window"],
            train=False,
            data_keys=[
                "x6d",
                "root",
                "offsets",
                "avg_speed_3d",
                # "raw_pose",
            ],
            normalize=[],
            shuffle=False,
        )

        spd_std = torch.tensor([0.5616, 0.3610, 0.4162])
        spd_mean = torch.tensor([0.6783, 0.7000, 0.6396])
        spd_true = (loader.dataset.data["avg_speed_3d"] - spd_mean) / spd_std

        # avg_speed_3d_rand = spd_true[torch.randperm(len(spd_true))]
        rand_jitter = torch.randn((spd_true.shape[0], 1)) * spd_std * 1.5 + 0.5

        avg_speed_3d_rand = spd_true + rand_jitter
        avg_speed_3d_rand = torch.clamp(
            avg_speed_3d_rand, min=spd_true.min(dim=0)[0], max=spd_true.max(dim=0)[0]
        )
        loader.dataset.data["avg_speed_3d"] = avg_speed_3d_rand
        # avg_speed_3d_rand = spd_true

        kinematic_tree = loader.dataset.kinematic_tree
        n_keypts = loader.dataset.n_keypts
        arena_size = loader.dataset.arena_size
        discrete_classes = loader.dataset.discrete_classes

        # loader.dataset.window_inds = loader.dataset.window_inds[::downsample]
    assert len(loader.dataset.window_inds) == len(spd_true)

    gs = subf[var_ind].add_gridspec(1, 4)

    if var_key == "heading":
        subf[var_ind].suptitle(
            "Generated {} For Random Input {}".format(titles[var_key], titles[var_key]),
            fontsize=14,
        )

    elif var_key == "avg_speed_3d":
        subf[var_ind].suptitle(
            "Generated {} For Random Input {}".format(titles[var_key], titles[var_key]),
            fontsize=14,
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

        # z = ssumo.get.latents(
        #     config,
        #     model,
        #     models[model_key][1],
        #     loader,
        #     device="cuda",
        #     dataset_label="Test",
        # )
        # z = torch.tensor(
        #     np.load(path + "latents/Test_{}.npy".format(models[model_key][1]))
        # )

        # assert len(z) == len(loader.dataset)

        # if var_key == "heading":
        #     z = torch.cat(
        #         [z, spd_true, heading2D_rand],
        #         dim=-1,
        #     )
        # elif var_key == "avg_speed_3d":
        #     if "heading" in model.conditional_keys:
        #         z = torch.cat(
        #             [z, avg_speed_3d_rand, heading_true],
        #             dim=-1,
        #         )
        #     else:
        #         z = torch.cat(
        #             [z, avg_speed_3d_rand],
        #             dim=-1,
        #         )

        # loader.dataset.data["z"] = z
        pose = []
        model.eval()
        for batch_idx, data in enumerate(tqdm.tqdm(loader)):
            data = {k: v.to("cuda") for k, v in data.items()}
            data_o = model.encode(data)
            data_o.update(model.decode(data_o["mu"], data))
            pose_batch = fwd_kin_cont6d_torch(
                data_o["x6d"].reshape((-1, n_keypts, 6)),
                kinematic_tree,
                data["offsets"].reshape((-1,) + data["offsets"].shape[-2:]),
                root_pos=data_o["root"].reshape(-1, 3),
                do_root_R=True,
                eps=1e-8,
            ).reshape((-1, model.window, n_keypts, 3))

            # x_hat = model.decoder(data["z"].cuda()).moveaxis(-1, 1)
            # x6d = x_hat[..., :-3].reshape(data["z"].shape[0], model.window, -1, 6)
            # offsets = data["offsets"].cuda()

            # if var_key == "heading":
            #     root = torch.zeros((data["z"].shape[0]* model.window, 3), device=x6d.device)
            # elif var_key == "avg_speed_3d":
            #     root = model.inv_normalize_root(x_hat[..., -3:]).reshape(
            #     data["z"].shape[0]* model.window, 3
            # )
            # pose_batch = fwd_kin_cont6d_torch(
            #     x6d.reshape((-1, n_keypts, 6)),
            #     kinematic_tree,
            #     offsets.reshape((-1,) + offsets.shape[-2:]),
            #     root_pos=root,
            #     do_root_R=True,
            #     eps=1e-8,
            # ).reshape((-1, model.window, n_keypts, 3))

            pose += [pose_batch.detach().cpu()]

        pose = torch.cat(pose, axis=0).numpy()

        if var_key == "heading":
            heading1D_out = get_frame_yaw(pose[:, model.window // 2, ...], 0, 1)[
                ..., None
            ]
            pred_out = get_angle2D(heading1D_out)
            r2 = r2_score(heading2D_rand, pred_out)
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
            # import pdb; pdb.set_trace()
            expected = (avg_speed_3d_rand * spd_std + spd_mean).mean(axis=1)
            pred_out = pred_out.mean(axis=-1)

        ax = subf[var_ind].add_subplot(gs[i])
        ax.set_title(model_key + "\n$R^2 = ${:3f}".format(r2))
        y_eq_x = np.linspace(
            min(expected.min(), pred_out.min()),
            max(expected.max(), pred_out.max()),
            1000,
        )
        ax.plot(y_eq_x, y_eq_x, label="$y=x$", c="k", lw=1)
        ax.scatter(
            expected, pred_out, s=3, marker="o", facecolors="none", edgecolors="C0"
        )
        ax.set_xlabel("Input {}".format(titles[var_key]))
        ax.set_ylabel("Generated {}".format(titles[var_key]))
        ax.legend(loc="lower right" if var_key == "avg_speed_3d" else "lower center")

    subf[var_ind].subplots_adjust(
        left=0.05, bottom=0.15, right=0.98, top=0.75, wspace=0.3, hspace=0.1
    )

plt.savefig("./results/genres_final_test.png", dpi=400)
