import numpy as np
import matplotlib.pyplot as plt
import ssumo
# from base_path import RESULTS_PATH, CODE_PATH
import sys
from pathlib import Path
from dappy import read
import torch

RESULTS_PATH = "/mnt/ceph/users/jwu10/results/vae/"
CODE_PATH = "/mnt/home/jwu10/working/ssumo/"
palette = ssumo.plot.constants.PALETTE_2
# task_id = sys.argv[1] if len(sys.argv)>1 else ""
experiment_folder = sys.argv[1]
epoch = 200
if experiment_folder.endswith(".yaml"):
    config_path = CODE_PATH + "/configs/" + experiment_folder
    config = read.config(config_path)
    analysis_keys = config["PATHS"]
    out_path = config["OUT"]
elif Path(RESULTS_PATH + experiment_folder).is_dir():
    analysis_keys = [experiment_folder]
    out_path = RESULTS_PATH + experiment_folder

config = read.config(RESULTS_PATH + analysis_keys[0] + "/model_config.yaml")
loader = ssumo.get.mouse_data(
    data_config=config["data"],
    window=config["model"]["window"],
    train=True,
    data_keys=[
        "x6d",
        "root",
        "avg_speed_3d",
    ],
    shuffle=False,)

f, ax = plt.subplots(3, 3, figsize=(15, 15))
for i, an_key in enumerate(analysis_keys):
    path = "{}/{}/".format(RESULTS_PATH, an_key)
    config = read.config(path + "/model_config.yaml")
    config["model"]["load_model"] = config["out_path"]

    model = ssumo.get.model(
                model_config=config["model"],
                load_model=config["out_path"],
                epoch=200,
                disentangle_config=config["disentangle"],
                n_keypts=loader.dataset.n_keypts,
                direction_process=config["data"]["direction_process"],
                loss_config = config["loss"],
                arena_size=loader.dataset.arena_size,
                kinematic_tree=loader.dataset.kinematic_tree,
                bound=config["data"]["normalize"] is not None,
                discrete_classes=loader.dataset.discrete_classes,
                verbose=-1,
            )

    z = ssumo.get.latents(config, model, epoch, loader, "cuda", "Train")
    z = torch.cat([z, loader.dataset[:]["avg_speed_3d"]], dim=-1)
    z -= z.mean(dim=0)
    # z /= z.std(dim=0)
    corr = (z.T @ z)/(len(z)-1)

    im = ax[i//3,i%3 ].imshow(torch.abs(corr), vmin=0, vmax=1)
    ax[i//3,i%3 ].set_title(an_key)

f.colorbar(im, ax=ax.ravel().tolist())
plt.savefig(out_path + "/z_corr.png")
plt.close()
