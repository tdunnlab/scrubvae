import ssumo
from torch.utils.data import DataLoader
import torch

torch.autograd.set_detect_anomaly(True)
import torch.optim as optim
import tqdm
from ssumo.params import read
import pickle
import sys
from base_path import RESULTS_PATH

### Set/Load Parameters
analysis_key = sys.argv[1]
print(analysis_key)
config = read.config(RESULTS_PATH + analysis_key + "/model_config.yaml")

### Load Dataset
dataset, loader = ssumo.data.get_mouse(
    data_config=config["data"],
    window=config["model"]["window"],
    train=True,
    data_keys=["x6d", "root", "offsets"] + config["disentangle"]["features"],
    shuffle=True,
)

# Balance disentanglement losses
if config["disentangle"]["balance_loss"]:
    print("Balancing disentanglement losses")
    for k in config["disentangle"]["features"]:
        std = dataset[:][k].std() * dataset[0][k].shape[-1]
        config["loss"][k] /= std
        if k + "_gr" in config["loss"].keys():
            config["loss"][k + "_gr"] /= std

vae, device = ssumo.model.get(
    model_config=config["model"],
    disentangle_config=config["disentangle"],
    n_keypts=dataset.n_keypts,
    direction_process=config["data"]["direction_process"],
    arena_size=dataset.arena_size,
    kinematic_tree=dataset.kinematic_tree,
    verbose=1,
)
optimizer = optim.Adam(vae.parameters(), lr=0.0001)

beta_schedule = ssumo.train.get_beta_schedule(
    config["loss"]["prior"],
    config["train"]["num_epochs"] - config["model"]["start_epoch"],
    config["train"]["beta_anneal"],
)

loss_dict_keys = ["total"] + list(config["loss"].keys())
loss_dict = {k: [] for k in loss_dict_keys}
avgtime = []
for epoch in tqdm.trange(
    config["model"]["start_epoch"] + 1, config["train"]["num_epochs"] + 1
):
    config["loss"]["prior"] = beta_schedule[epoch - config["model"]["start_epoch"] - 1]
    print("Beta schedule: {}".format(config["loss"]["prior"]))

    # epoch_loss, times = ssumo.train.train_epoch(
    epoch_loss = ssumo.train.train_epoch(
        vae,
        optimizer,
        loader,
        device,
        config["loss"],
        epoch,
        mode="train",
        disentangle_keys=config["disentangle"]["features"],
        timer=False,
    )
    # avgtime.append(times)
    loss_dict = {k: v + [epoch_loss[k]] for k, v in loss_dict.items()}

    if epoch % 10 == 0:
        # print(
        #     "Average time taken to move to gpu, forward pass, loss calc, gradient calc, optimizer step, and total:\n"
        # )
        # [print(i) for i in torch.mean(torch.Tensor(avgtime), 0)]
        print("Saving model to folder: {}".format(config["out_path"]))
        torch.save(
            {k: v.cpu() for k, v in vae.state_dict().items()},
            "{}/weights/epoch_{}.pth".format(config["out_path"], epoch),
        )
        # timename = ["to_gpu", "forward", "loss", "gradient", "opt_step"]
        # torch.save(
        #     {timename[i]: avgtime for i in range(5)},
        #     "{}/weights/epoch_{}.pth".format(config["out_path"], epoch),
        # )

        pickle.dump(
            loss_dict,
            open("{}/losses/loss_dict.pth".format(config["out_path"]), "wb"),
        )

        ssumo.plot.eval.loss(
            loss_dict, config["out_path"], config["disentangle"]["features"]
        )
