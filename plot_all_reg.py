import numpy as np
import utils
from dappy import read
from data.dataset import MouseDatasetOld
import matplotlib.pyplot as plt
from sklearn.metrics import r2_score
import scipy.linalg as spl
from sklearn.linear_model import LinearRegression
from pathlib import Path
import re

palette = [
    "#e60049",
    "#0bb4ff",
    "#50e991",
    "#e6d800",
    "#9b19f5",
    "#ffa300",
    "#dc0ab4",
    "#b3d4ff",
    "#00bfa0",
    "#b30000",
    "#7c1158",
    "#4421af",
]

# out_base = "/mnt/home/jwu10/working/ceph/results/vae/"
base_path = "/mnt/ceph/users/jwu10/results/vae/gr_parts1/"
# paths = ["rc_w51_midfwd_full","rc_w51_ba_midfwd_full"]

paths = ["partspd10_gre1_rc_w51_b1_midfwd_full", "partspd10_rc_w51_b1_midfwd_full"]

# paths = [
    # "rc_w51_midfwd_full",
    # "rc_w51_ba_midfwd_full",
    # "avgspd_rc_w51_b1_midfwd_full",
    # "avgspd_rc_w51_midfwd_full",
    # "avgspd_ndgre20_rc_w51_b1_midfwd_full_a1",
    # "avgspd_ndgre_rc_w51_b1_midfwd_full_a05",
    # "avgspd_ndgre_rc_w51_midfwd_full_a1",
    # "avgspd_ndgre1_rc_w51_b1_midfwd_full_a05",
    # "avgspd_gre_rc_w51_midfwd_full_a1",
    # "avgspd_gre_rc_w51_ba_midfwd_full_a1",
    # "avgspd_gre_rc_w51_b1_midfwd_full_a05",
    # "avgspd_gre1_rc_w51_b1_midfwd_full_a05",
# ]

f, ax_arr = plt.subplots(2, 1, figsize=(15, 15))
for path_ind, path in enumerate(paths):
    config = read.config(base_path + path + "/model_config.yaml")
    config["load_model"] = config["out_path"]

    # Get all epochs
    z_path = Path(config["out_path"] + "weights/")
    epochs = [re.findall(r"\d+", f.parts[-1]) for f in list(z_path.glob("epoch*"))]
    epochs = np.sort(np.array(epochs).astype(int).squeeze())
    # epochs = epochs[np.where(epochs<201)[0]]
    print("Epochs found: {}".format(epochs))

    if path == paths[0]:
        dataset = MouseDatasetOld(
            data_path=config["data_path"],
            skeleton_path="/mnt/home/jwu10/working/behavior_vae/configs/mouse_skeleton.yaml",
            train=True,
            window=config["window"],
            stride=1,
            direction_process=config["direction_process"],
            get_speed=True,
            arena_size=config["arena_size"],
            invariant=config["invariant"],
            get_raw_pose=False,
        )
        spd_true = dataset[:]["speed"].mean(dim=-1, keepdim=True).detach().cpu().numpy()

    metrics = {
        "R2": [],
        "R2_Sub": [],
        "w_Norm": [],
        "w_Norm_Sub": [],
        "Bias": [],
        "Bias_Sub": [],
    }
    for epoch in epochs:
        config["load_epoch"] = epoch

        if config["speed_decoder"] is None:
            vae, device = utils.init_model(
                config, dataset.n_keypts, config["invariant"], verbose=-1
            )
        else:
            vae, spd_decoder, device = utils.init_model(
                config,
                dataset.n_keypts,
                config["invariant"],
                verbose=-1,
            )
            spd_decoder.eval()
        vae.eval()

        z = utils.get_latents(vae, dataset, config, device, "Train")
        lin_model = LinearRegression().fit(z, spd_true)
        spd_pred = lin_model.predict(z)

        metrics["R2"] += [r2_score(spd_true, spd_pred)]
        metrics["w_Norm"] += [(lin_model.coef_ @ lin_model.coef_.T)[0, 0]]
        metrics["Bias"] += [lin_model.intercept_[0]]
        print(metrics["R2"])

        if config["speed_decoder"] is not None:
            if config["gradient_reversal"]:
                spd_weights = spd_decoder.decoder.weight.cpu().detach().numpy()
            else:
                spd_weights = spd_decoder.weight.cpu().detach().numpy()
        else:
            spd_weights = lin_model.coef_
            z -= lin_model.intercept_ * spd_weights

        # nrm = (spd_weights @ spd_weights.T).ravel()
        # avg_spd_o = z @ spd_weights.T
        # z_sub = z - (avg_spd_o @ spd_weights) / nrm
        U_orth = spl.null_space(spd_weights)
        z_sub = z @ U_orth
        lin_model_sub = LinearRegression().fit(z_sub, spd_true)
        spd_pred_sub = lin_model_sub.predict(z_sub)

        metrics["R2_Sub"] += [r2_score(spd_true, spd_pred_sub)]
        metrics["w_Norm_Sub"] += [(lin_model_sub.coef_ @ lin_model_sub.coef_.T)[0, 0]]
        metrics["Bias_Sub"] += [lin_model_sub.intercept_[0]]
        print(metrics["R2_Sub"])

    # if config["speed_decoder"] is None:
    #     ax_arr[1].plot(epochs, metrics["R2"], label="{}_Sup".format(path))

    for i, key in enumerate(["R2", "R2_Sub"]):  # , "w_Norm", "w_Norm_Sub"]):
        if "Norm" in key:
            ax_arr[i].plot(epochs, np.log10(metrics[key]), label="{}".format(path))
        else:
            ax_arr[i].plot(
                epochs,
                metrics[key],
                label="{}".format(path),
                color=palette[path_ind],
                alpha=0.5,
            )
        ax_arr[i].set_ylabel(key)
        # ax_arr[1].set_ylim(bottom=0.2, top=1)
        ax_arr[i].legend()
        ax_arr[i].set_xlabel("Epoch")

    f.tight_layout()

    plt.savefig(base_path + "/spd_reg_epoch.png")
