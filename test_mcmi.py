import ssumo
import torch

torch.autograd.set_detect_anomaly(True)
import torch.optim as optim
import tqdm
from ssumo.params import read
import pickle
import sys
from base_path import RESULTS_PATH
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
import numpy as np
from ssumo.train.mutual_inf import MutInfoEstimator

torch.manual_seed(50)

### Set/Load Parameters
analysis_key = sys.argv[1]
config = read.config(RESULTS_PATH + analysis_key + "/model_config.yaml")
config["model"]["load_model"] = config["out_path"]
config["data"]["batch_size"] = 4096
config["data"]["stride"] = 1
disentangle_keys = ["avg_speed_3d", "heading", "heading_change"]
### Load Dataset
dataset, loader = ssumo.data.get_mouse(
    data_config=config["data"],
    window=config["model"]["window"],
    train=True,
    data_keys=["x6d", "root"] + disentangle_keys,
    shuffle=False,
    normalize=disentangle_keys,
)

epochs_to_test = ssumo.eval.metrics.get_all_epochs(config["out_path"])
mcmi = []
for epoch_ind, epoch in enumerate(epochs_to_test):
    config["model"]["start_epoch"] = epoch
    model, device = ssumo.model.get(
        model_config=config["model"],
        disentangle_config=config["disentangle"],
        n_keypts=dataset.n_keypts,
        direction_process=config["data"]["direction_process"],
        arena_size=dataset.arena_size,
        kinematic_tree=dataset.kinematic_tree,
        verbose=-1,
    )
    model.eval()

    dataset.data["mu"] = ssumo.eval.get.latents(model, dataset, config, device, "Train")

    loader_mc = DataLoader(
        dataset=dataset,
        batch_size=4096,
        shuffle=True,
    )
    data_mc = next(iter(loader_mc))

    data_mco = model.encode({k: data_mc[k].to(device) for k in ["x6d", "root"]})
    mu_mc, L_mc = [d.detach() for d in data_mco]
    var_mc = L_mc.diagonal(dim1=-2, dim2=-1)**2 + 1
    v_mc = torch.cat([data_mc[k] for k in disentangle_keys], dim=-1).to(device)

    mi_estimator = MutInfoEstimator(
        mu_mc,
        v_mc,
        var_mc,
        gamma=1,
        var_mode="diagonal",
        device=device,
    )
    mcmi_temp = 0
    for batch_idx, data in enumerate(tqdm.tqdm(loader)):
        v = torch.cat([data[k] for k in disentangle_keys], dim=-1).to(device)
        z = data["mu"].cuda()

        mcmi_temp += mi_estimator(z,v).item() / len(loader)

    mcmi += [mcmi_temp]
    print(mcmi)

    pickle.dump(
        mcmi,
        open("{}/mcmi.p".format(config["out_path"]), "wb"),
    )

    # mcmi = pickle.load(open("{}/mcmi.p".format(config["out_path"]), "rb"))

    f = plt.figure(figsize=(10, 5))
    plt.plot(
        epochs_to_test[: epoch_ind + 1],
        mcmi,
    )
    # import pdb; pdb.set_trace()
    plt.xlabel("Epoch")
    plt.ylabel("MI")
    plt.legend()
    plt.savefig("{}/mutual_inf.png".format(config["out_path"]))
