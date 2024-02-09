import ssumo
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

if len(sys.argv)> 2:
    job_id = sys.argv[2]
    print(job_id)
    print(sys.argv)
    analysis_key = "{}/{}/".format(analysis_key, job_id)

config = read.config(RESULTS_PATH + analysis_key + "/model_config.yaml")

### Load Dataset
dataset, loader = ssumo.data.get_mouse(
    data_config=config["data"],
    window=config["model"]["window"],
    train=True,
    data_keys=["x6d", "root", "offsets", "target_pose"] + config["disentangle"]["features"],
    shuffle=True,
)

#Balance disentanglement losses
if config["disentangle"]["balance_loss"]:
    print("Balancing disentanglement losses")
    for k in config["disentangle"]["features"]:
        var = torch.sqrt((dataset[:][k].std(dim=0)**2).sum()).detach().numpy()
        config["loss"][k] /= var
        if k + "_gr" in config["loss"].keys():
            config["loss"][k+"_gr"] /=var

    print("Finished disentanglement loss balancing...")
    print(config["loss"])

vae, device = ssumo.model.get(
    model_config=config["model"],
    disentangle_config=config["disentangle"],
    n_keypts = dataset.n_keypts,
    direction_process=config["data"]["direction_process"],
    arena_size = dataset.arena_size,
    kinematic_tree = dataset.kinematic_tree,
    verbose=1,
)

optimizer = optim.AdamW(vae.parameters(), lr=config["train"]["lr"])
scheduler = optim.lr_scheduler.CosineAnnealingWarmRestarts(optimizer, T_0 = 50)

beta_schedule = ssumo.train.get_beta_schedule(
    config["loss"]["prior"],
    config["train"]["num_epochs"] - config["model"]["start_epoch"],
    config["train"]["beta_anneal"],
)

loss_dict_keys = ["total"] + list(config["loss"].keys())
loss_dict = {k: [] for k in loss_dict_keys}

if device == "cuda":
    torch.backends.cudnn.benchmark = True

for epoch in tqdm.trange(
    config["model"]["start_epoch"] + 1, config["train"]["num_epochs"] + 1
):
    config["loss"]["prior"] = beta_schedule[epoch - config["model"]["start_epoch"] - 1]
    print("Beta schedule: {:.3f}".format(config["loss"]["prior"]))

    epoch_loss = ssumo.train.train_epoch(
        vae,
        optimizer,
        scheduler,
        loader,
        device,
        config["loss"],
        epoch,
        mode="train",
        disentangle_keys=config["disentangle"]["features"],
    )
    loss_dict = {k: v + [epoch_loss[k]] for k,v in loss_dict.items()}

    if epoch % 10 == 0:
        print("Saving model to folder: {}".format(config["out_path"]))
        torch.save(
            {k: v.cpu() for k, v in vae.state_dict().items()},
            "{}/weights/epoch_{}.pth".format(config["out_path"], epoch),
        )

        pickle.dump(
            loss_dict,
            open("{}/losses/loss_dict.pth".format(config["out_path"]), "wb"),
        )

        ssumo.plot.eval.loss( loss_dict, config["out_path"], config["disentangle"]["features"] )
