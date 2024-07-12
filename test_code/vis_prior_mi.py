import matplotlib.pyplot as plt
from neuroposelib import read
from pathlib import Path
import numpy as np
import pickle
import sys

RESULTS_PATH = "/mnt/ceph/users/jwu10/results/vae/"

experiment_folder = "mi_64_bw"
losses = {"500": [], "750": []}
bw = {k: np.arange(250, 1751, 125, dtype=int) for k in losses.keys()}

for i in np.arange(250, 1751, 125, dtype=int):
    for key in losses.keys():
        try:
            loss_dict = pickle.load(
                open(
                    "{}/{}/{}_{}/losses/loss_dict_Train.p".format(
                        RESULTS_PATH, experiment_folder, i, key
                    ),
                    "rb",
                )
            )
            epochs = np.array(loss_dict["epoch"])
            if len(np.where((epochs > 200) & (epochs < 400))[0]) != 0:
                losses[key] += [
                    np.array(loss_dict["prior"])[
                        np.where((epochs > 200) & (epochs < 400))[0]
                    ].mean()
                ]
            else:
                bw[key] = np.delete(bw[key], np.where(bw[key] == i)[0])
        except:
            bw[key] = np.delete(bw[key], np.where(bw[key] == i)[0])

plt.figure(figsize=(10, 5))
plt.title("Prior loss for MI Models")
for k, v in losses.items():
    print(bw[k])
    plt.plot(
        bw[k] / 1000,
        v,
        label="Weight: {}".format(k),
        alpha=0.5,
        linewidth=1,
    )

baselines = {"Vanilla": "vanilla_64/1", "C-VAE": "cvae_64/1"}

for model, path in baselines.items():
    loss_dict = pickle.load(
        open(
            "{}/{}/losses/loss_dict_Train.p".format(RESULTS_PATH, path), "rb"
        )
    )
    epochs = np.array(loss_dict["epoch"])
    mean = np.array(loss_dict["prior"])[
                        np.where((epochs > 200) & (epochs < 400))[0]
                    ].mean()
                
    plt.hlines(mean,xmin=bw["500"].min()/1000,xmax=bw["500"].max()/1000, label=model)

plt.yscale("log")
plt.xlabel("Bandwidth")
plt.ylabel("Log Loss")
plt.legend()
plt.savefig("/mnt/ceph/users/jwu10/results/vae/" + experiment_folder + "/prior.png")
plt.close()
