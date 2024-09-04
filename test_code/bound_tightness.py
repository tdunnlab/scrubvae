import numpy as np
import matplotlib.pyplot as plt
import ssumo
import pickle
from scipy.spatial.distance import cdist, pdist
import matplotlib.font_manager
import matplotlib.transforms as mtrans

# from base_path import RESULTS_PATH, CODE_PATH
import sys
from pathlib import Path
from neuroposelib import read
from sklearn.decomposition import PCA
import colorcet as cc
from scipy.stats import circvar
import tqdm
from ssumo.eval.metrics import linear_rand_cv, mlp_rand_cv
from sklearn.model_selection import KFold
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score
import torch

CODE_PATH = "/mnt/home/jwu10/working/ssumo/"
RESULTS_PATH = "/mnt/ceph/users/jwu10/results/vae/"

window=51
for var_ind, var_key in enumerate(["heading", "avg_speed_3d"]):
    print(var_key)
    models = read.config(CODE_PATH + "configs/exp_lqc.yaml")[var_key]
    models = {m[0]: [m[1], m[2]] for m in models if m[0] not in ["VAE", "beta-VAE", "C-VAE"]}

    if var_ind == 0:
        config = read.config(RESULTS_PATH + models["SC-VAE-MALS"][0] + "/model_config.yaml")
        loader = ssumo.get.mouse_data(
            data_config=config["data"],
            window=config["model"]["window"],
            train=True,
            data_keys=["x6d", "root", "avg_speed_3d", "heading", "ids"],
            shuffle=False,
        )

    y_true = loader.dataset[:][var_key].cpu().numpy()

    for model_key in models.keys():
        path = "{}{}/".format(RESULTS_PATH, models[model_key][0])
        config = read.config(path + "/model_config.yaml")
        model = ssumo.get.model(
            model_config=config["model"],
            load_model=config["out_path"],
            epoch=models[model_key][1],
            disentangle_config=config["disentangle"],
            n_keypts=loader.dataset.n_keypts,
            direction_process=config["data"]["direction_process"],
            loss_config=config["loss"],
            arena_size=loader.dataset.arena_size,
            kinematic_tree=loader.dataset.kinematic_tree,
            bound=config["data"]["normalize"] is not None,
            discrete_classes=loader.dataset.discrete_classes,
            verbose=-2,
        )

        path = "{}{}/".format(RESULTS_PATH, models[model_key][0])
        z = torch.tensor(np.load(path + "latents/Train_{}.npy".format(models[model_key][1])))

        y_pred = LinearRegression().fit(z, y_true).predict(z)
        print("Linear {} R2 for {}: {}".format(var_key, model_key, r2_score(y_true, y_pred)))

        y_pred = ssumo.eval.metrics.train_MLP(z, y_true)[1]
        print("MLP {} R2 for {}: {}".format(var_key, model_key, r2_score(y_true, y_pred)))
        method = list(model.disentangle.keys())[0]
        if "MACS" not in model_key:
            y_pred = model.disentangle[method][var_key](z.cuda())
            y_pred = torch.cat([y[None,...] for y in y_pred], dim=0).mean(0).cpu().detach().numpy()
        else:
            # import pdb; pdb.set_trace()
            loader.dataset.data["z"] = z
            y_pred = []
            for batch_idx, data in enumerate(tqdm.tqdm(loader)):
                z_batch = data["z"].cuda()
                y_pred_batch = model.disentangle[method][var_key](z_batch)
                y_pred += [torch.cat([y[None,...] for y in y_pred_batch], dim=0).mean(0).cpu().detach()]

            y_pred = torch.cat(y_pred, dim=0)

        print("{} {} R2 for {}: {}".format(method,var_key, model_key, r2_score(y_true, y_pred)))
