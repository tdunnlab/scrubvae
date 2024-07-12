import ssumo
from torch.utils.data import DataLoader
from neuroposelib import read
import matplotlib.pyplot as plt
import torch
from neuroposelib import visualization as vis
import numpy as np
from pathlib import Path
import sys
import tqdm
RESULTS_PATH = "/mnt/ceph/users/jwu10/results/vae/"

analysis_key = "beta_prior/vanilla"
config = read.config(RESULTS_PATH + analysis_key + "/model_config.yaml")
connectivity = read.connectivity_config(config["data"]["skeleton_path"])

dataset_label = "Test"
config["data"]["stride"] = 2
### Load Datasets
jpe = {"mu":[], "mode":[]}
root_mse = {"mu":[], "mode":[]}
epochs = np.arange(250, 330, 10, dtype=int)
for epoch in epochs:
    loader, model = ssumo.get.data_and_model(
        config,
        load_model=config["out_path"],
        epoch=epoch,
        dataset_label=dataset_label,
        data_keys=["x6d", "root", "offsets", "target_pose"],
        shuffle=False,
        verbose=0,
    )
    device = "cuda"
    model.eval()
    mu_jpe, mode_jpe = 0, 0
    mu_root_mse, mode_root_mse = 0, 0
    for batch_idx, data in enumerate(tqdm.tqdm(loader)):
        data = {k: v.to(device) for k, v in data.items()}
        data_o = model.encode(data)
        mu = data_o["mu"]
        x_mu = model.decoder(mu).moveaxis(-1, 1)
        x6d_mu = x_mu[..., :-3].reshape(mu.shape[0], model.window, -1, 6)
        root_mu = model.inv_normalize_root(x_mu[..., -3:]).reshape(
            mu.shape[0], model.window, 3
        )
        mu_jpe += ssumo.train.losses.mpjpe_loss(data["target_pose"], x6d_mu, model.kinematic_tree, data["offsets"]).cpu().detach().numpy()
        mu_root_mse += (torch.nn.MSELoss(reduction="sum")(root_mu, data["root"]) / config["data"]["batch_size"]).cpu().detach().numpy()

        mode = ((data_o["alpha"] - 1) / (data_o["alpha"] + data_o["beta"] - 2)) * 2 - 1
        x_mode = model.decoder(mode).moveaxis(-1, 1)
        x6d_mode = x_mode[..., :-3].reshape(mode.shape[0], model.window, -1, 6)
        root_mode = model.inv_normalize_root(x_mode[..., -3:]).reshape(
            mode.shape[0], model.window, 3
        )
        mode_jpe += ssumo.train.losses.mpjpe_loss(data["target_pose"], x6d_mode, model.kinematic_tree, data["offsets"]).cpu().detach().numpy()
        mode_root_mse += (torch.nn.MSELoss(reduction="sum")(root_mode, data["root"]) / config["data"]["batch_size"]).cpu().detach().numpy()

    jpe["mu"] += [mu_jpe/len(loader)]
    root_mse["mu"] += [mu_root_mse/len(loader)]

    jpe["mode"] += [mode_jpe/len(loader)]
    root_mse["mode"] += [mode_root_mse/len(loader)]


f = plt.figure(figsize=(15, 10))

for key in ["mu", "mode"]:
    plt.plot(epochs, jpe[key], label="{}".format(key))

plt.legend()
plt.xlabel("epoch")
plt.ylabel("JPE")
plt.savefig(RESULTS_PATH + analysis_key + "/jpe.png")
plt.close()

f = plt.figure(figsize=(15, 10))
for key in ["mu", "mode"]:
    plt.plot(epochs, root_mse[key], label="Root {}".format(key))

plt.legend()
plt.xlabel("epoch")
plt.ylabel("JPE")
plt.savefig(RESULTS_PATH + analysis_key + "/root.png")
plt.close()