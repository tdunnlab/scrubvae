import numpy as np
from dappy import read
# from ssumo.data.dataset import MouseDatasetOld
import matplotlib.pyplot as plt
from sklearn.metrics import r2_score
import scipy.linalg as spl
from sklearn.linear_model import LinearRegression
from pathlib import Path
import re
import ssumo

base_path = "/mnt/ceph/users/jwu10/results/vae/heading/"
paths = ["gre1_b1_true_x360", "vanilla", "no_gr"]

f, ax_arr = plt.subplots(2, 1, figsize=(15, 15))
for path_ind, path in enumerate(paths):
    config = read.config(base_path + path + "/model_config.yaml")
    config["model"]["load_model"] = config["out_path"]

    # Get all epochs
    z_path = Path(config["out_path"] + "weights/")
    epochs = [re.findall(r"\d+", f.parts[-1]) for f in list(z_path.glob("epoch*"))]
    epochs = np.sort(np.array(epochs).astype(int).squeeze())
    print("Epochs found: {}".format(epochs))

    if path == paths[0]:
        dataset = ssumo.data.get_mouse(
            data_config=config["data"],
            window=config["model"]["window"],
            train="Train",
            data_keys=["x6d", "root"]
            + config["disentangle"]["features"],
        )

    metrics = {
        "R2": [],
        "R2_Sub": []
    }
    for epoch in epochs:
        config["model"]["start_epoch"] = epoch

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
