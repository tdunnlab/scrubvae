import numpy as np
from dappy import read
import matplotlib.pyplot as plt
from sklearn.metrics import r2_score
import scipy.linalg as spl
from sklearn.linear_model import LinearRegression
from pathlib import Path
import re
import ssumo
import pickle
from base_path import RESULTS_PATH

palette = ssumo.plot.constants.PALETTE_2
results_path = RESULTS_PATH + "/heading/"
paths = ["gre1_b1_true_x360", "balanced", "no_gr", "bal_hc_sum", "vanilla"]
dataset_label = "Train"

for path_ind, path in enumerate(paths):
    config = read.config(results_path + path + "/model_config.yaml")
    config["model"]["load_model"] = config["out_path"]

    # Get all epochs
    epochs = ssumo.eval.metrics.get_all_epochs(config["out_path"])

    if path == paths[0]:
        disentangle_keys = config["disentangle"]["features"]
        dataset, loader = ssumo.data.get_mouse(
            data_config=config["data"],
            window=config["model"]["window"],
            train=dataset_label == "Train",
            data_keys=[
                "x6d",
                "root",
            ]
            + disentangle_keys,
            shuffle = False,
        )
        met_dict = {
            k: {"R2": [], "R2_Null": []} for k in disentangle_keys
        }
        met_dict["epochs"] = []
        metrics = {p: met_dict.copy() for p in paths}

    metrics[path]["epochs"] = epochs

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

        for key in disentangle_keys:
            print("Decoding Feature: {}".format(key))
            y_true = dataset[:][key].detach().cpu().numpy()
            lin_model = LinearRegression().fit(z, y_true)
            pred = lin_model.predict(z)
            print(metrics)

            metrics[path][key]["R2"] += [r2_score(y_true, pred)]
            print(metrics[path][key]["R2"])

            if len(vae.disentangle.keys()) > 0:
                dis_w = vae.disentangle[key].decoder.weight.detach().cpu().numpy()
            else:
                dis_w = lin_model.coef_
                # z -= lin_model.intercept_[:,None] * dis_w

            ## Null space projection
            z_null = ssumo.eval.project_to_null(z, dis_w)[0]
            pred_null = LinearRegression().fit(z_null, y_true).predict(z_null)

            metrics[path][key]["R2_Null"] += [r2_score(y_true, pred_null)]
            print(metrics[path][key]["R2_Null"])

    pickle.dump(metrics, open("{}/linreg.p".format(results_path), "wb"))

## Plot R^2
for key in disentangle_keys:
    f, ax_arr = plt.subplots(2, 1, figsize=(15, 15))
    for path_i, p in enumerate(paths):
        for i, metric in enumerate(["R2", "R2_Null"]):
            if "Norm" in metric:
                ax_arr[i].plot(
                    metrics[p]["epochs"],
                    np.log10(metrics[p][key][metric]),
                    label="{}".format(p),
                )
            else:
                ax_arr[i].plot(
                    metrics[p]["epochs"],
                    metrics[p][key][metric],
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
