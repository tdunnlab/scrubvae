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

palette = ssumo.plot.constants.PALETTE_2

base_path = "/mnt/ceph/users/jwu10/results/vae/heading/"
paths = ["vanilla","gre1_b1_true_x360", "balanced", "no_gr", "bal_hc_sum"]

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
            train="Test",
            data_keys=["x6d", "root", "heading"],# + config["disentangle"]["features"],
        )
        loader = DataLoader(
            dataset=dataset, batch_size=config["train"]["batch_size"], shuffle=False
        )

    metrics = {"R2": [], "R2_Sub": []}
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

        z = ssumo.evaluate.get.latents(vae, dataset, config, device, "Test")

        for key in ["heading"]:  # config["disentangle"]["features"]:
            y_true = dataset[:][key].detach().cpu().numpy()
            lin_model = LinearRegression().fit(z, y_true)
            pred = lin_model.predict(z)

            metrics["R2"] += [r2_score(y_true, pred)]
            # metrics["w_Norm"] += [(lin_model.coef_ @ lin_model.coef_.T)[0, 0]]
            # metrics["Bias"] += [lin_model.intercept_[0]]
            print(metrics["R2"])

            if len(vae.disentangle.keys())>0:
                dis_w = vae.disentangle[key].decoder.weight.detach().cpu().numpy()
            else:
                # import pdb; pdb.set_trace()
                dis_w = lin_model.coef_
                # z -= lin_model.intercept_[:,None] * dis_w

            ## Null space projection
            U_orth = spl.null_space(dis_w)
            z_sub = z @ U_orth
            lin_model_sub = LinearRegression().fit(z_sub, y_true)
            spd_pred_sub = lin_model_sub.predict(z_sub)

            metrics["R2_Sub"] += [r2_score(y_true, spd_pred_sub)]
            # metrics["w_Norm_Sub"] += [(lin_model_sub.coef_ @ lin_model_sub.coef_.T)[0, 0]]
            # metrics["Bias_Sub"] += [lin_model_sub.intercept_[0]]
            print(metrics["R2_Sub"])

        # if config["speed_decoder"] is None:
        #     ax_arr[1].plot(epochs, metrics["R2"], label="{}_Sup".format(path))

    for key in ["heading"]:
        for i, metric in enumerate(["R2", "R2_Sub"]):  # , "w_Norm", "w_Norm_Sub"]):
            if "Norm" in metric:
                ax_arr[i].plot(
                    epochs[: epoch_ind + 1],
                    np.log10(metrics[metric]),
                    label="{}".format(path),
                )
            else:
                ax_arr[i].plot(
                    epochs[: epoch_ind + 1],
                    metrics[metric],
                    label="{}".format(path),
                    color=palette[path_ind],
                    alpha=0.5,
                )

            ax_arr[i].set_ylabel(metric)
            ax_arr[i].legend()
            ax_arr[i].set_xlabel("Epoch")

        f.tight_layout()

        plt.savefig(base_path + "/{}_reg_epoch.png".format(key))
