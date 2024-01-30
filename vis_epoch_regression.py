import numpy as np
from dappy import read
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
from sklearn.metrics import r2_score
import scipy.linalg as spl
from sklearn.linear_model import LinearRegression
from pathlib import Path
import re
import ssumo

from base_path import RESULTS_PATH

palette = ssumo.plot.constants.PALETTE_2
results_path = RESULTS_PATH + "/heading/"
paths = ["gre1_b1_true_x360", "balanced", "no_gr", "bal_hc_sum", "vanilla"]
dataset_label = "Train"

for path_ind, path in enumerate(paths):
    config = read.config(results_path + path + "/model_config.yaml")
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
            train=dataset_label == "Train",
            data_keys=[
                "x6d",
                "root",
            ]
            + config["disentangle"]["features"],
        )
        loader = DataLoader(
            dataset=dataset, batch_size=config["train"]["batch_size"], shuffle=False
        )

    metrics = {k: {"R2": [], "R2_Sub": []} for k in config["disentangle"]["features"]}
    for epoch_ind, epoch in enumerate(epochs):
        config["model"]["start_epoch"] = epoch

        vae, device = ssumo.model.get(
            model_config=config["model"],
            disentangle_config=config["disentangle"],
            n_keypts=dataset.n_keypts,
            direction_process=config["data"]["direction_process"],
            arena_size=dataset.arena_size,
            kinematic_tree=dataset.kinematic_tree,
            verbose=-1,
        )

        z = ssumo.eval.get.latents(vae, dataset, config, device, dataset_label)

        for key in config["disentangle"]["features"]:
            print("Decoding Feature: {}".format(key))
            y_true = dataset[:][key].detach().cpu().numpy()
            lin_model = LinearRegression().fit(z, y_true)
            pred = lin_model.predict(z)

            metrics[key]["R2"] += [r2_score(y_true, pred)]
            # metrics["w_Norm"] += [(lin_model.coef_ @ lin_model.coef_.T)[0, 0]]
            # metrics["Bias"] += [lin_model.intercept_[0]]
            print(metrics[key]["R2"])

            if len(vae.disentangle.keys()) > 0:
                dis_w = vae.disentangle[key].decoder.weight.detach().cpu().numpy()
            else:
                dis_w = lin_model.coef_
                # z -= lin_model.intercept_[:,None] * dis_w

            ## Null space projection
            U_orth = spl.null_space(dis_w)
            z_sub = z @ U_orth
            lin_model_sub = LinearRegression().fit(z_sub, y_true)
            spd_pred_sub = lin_model_sub.predict(z_sub)

            metrics[key]["R2_Sub"] += [r2_score(y_true, spd_pred_sub)]
            # metrics["w_Norm_Sub"] += [(lin_model_sub.coef_ @ lin_model_sub.coef_.T)[0, 0]]
            # metrics["Bias_Sub"] += [lin_model_sub.intercept_[0]]
            print(metrics[key]["R2_Sub"])

        # if config["speed_decoder"] is None:
        #     ax_arr[1].plot(epochs, metrics["R2"], label="{}_Sup".format(path))

    for key in config["disentangle"]["features"]:
        f, ax_arr = plt.subplots(2, 1, figsize=(15, 15))
        for path_i, p in enumerate(paths[:path_ind]):
            # Get all epochs
            z_path = Path(results_path + p + "/weights/")
            epochs = [re.findall(r"\d+", f.parts[-1]) for f in list(z_path.glob("epoch*"))]
            epochs = np.sort(np.array(epochs).astype(int).squeeze())
            for i, metric in enumerate(["R2", "R2_Sub"]):  # , "w_Norm", "w_Norm_Sub"]):
                if "Norm" in metric:
                    ax_arr[i].plot(
                        epochs,
                        np.log10(metrics[key][metric]),
                        label="{}".format(p),
                    )
                else:
                    ax_arr[i].plot(
                        epochs,
                        metrics[key][metric],
                        label="{}".format(p),
                        color=palette[path_i],
                        alpha=0.5,
                    )

                ax_arr[i].set_ylabel(metric)
                ax_arr[i].legend()
                ax_arr[i].set_xlabel("Epoch")

        f.tight_layout()
        plt.savefig(results_path + "/{}_reg_epoch.png".format(key))
        plt.close()

