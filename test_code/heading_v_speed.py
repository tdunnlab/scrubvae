import numpy as np
from ssumo.eval import cluster
from neuroposelib import read
import torch
import ssumo

CODE_PATH = "/mnt/home/jwu10/working/ssumo/"
RESULTS_PATH = "/mnt/ceph/users/jwu10/results/vae/"

models = read.config(CODE_PATH + "configs/exp_finals.yaml")["avg_speed_3d"]
models = {m[0]: [m[1], m[2]] for m in models}
walking_list = [1, 4, 8, 38, 41, 44]

config = read.config(RESULTS_PATH + models["SC-VAE-MI"][0] + "/model_config.yaml")

loader = ssumo.get.mouse_data(
    data_config=config["data"],
    window=config["model"]["window"],
    train=True,
    data_keys=[
        "x6d",
        "root",
        "offsets",
        "avg_speed_3d",
        "heading"
    ],
    normalize=[],
    shuffle=False,
)

avg_speed_3d = loader.dataset[:]["avg_speed_3d"].cpu().detach().numpy()

heading = loader.dataset[:]["heading"].cpu().detach().numpy()
heading= np.arctan2(heading[:, 0], heading[:, 1])
heading_binned =np.digitize(heading, bins=np.linspace(-np.pi-0.1, np.pi+0.1, 25))

for i in range(3):
    ssumo.plot.feature_ridge(
            feature=avg_speed_3d[:,i],
            labels=heading_binned,
            xlabel="Avg Speed {}".format(i),
            ylabel="Heading Bin",
            x_lim=(avg_speed_3d[:,i].min() - 0.1, avg_speed_3d[:,i].max() + 0.1),
            n_bins=200,
            binrange=(avg_speed_3d[:,i].min(), avg_speed_3d[:,i].max()),
            path="./results/speed{}_v_heading".format(i),
        )
