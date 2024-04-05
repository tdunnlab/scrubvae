import ssumo
import torch
import functools

torch.autograd.set_detect_anomaly(True)
import tqdm
from ssumo.params import read
import pickle
import sys
from base_path import RESULTS_PATH
torch.backends.cudnn.benchmark = True

### Set/Load Parameters
analysis_key = sys.argv[1]

if len(sys.argv) > 2:
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
    data_keys=["x6d", "root", "offsets", "target_pose"]
    + config["disentangle"]["features"],
    shuffle=True,
    normalize=config["disentangle"]["features"],
)

model = ssumo.model.get(
    model_config=config["model"],
    disentangle_config=config["disentangle"],
    n_keypts=dataset.n_keypts,
    direction_process=config["data"]["direction_process"],
    arena_size=dataset.arena_size,
    kinematic_tree=dataset.kinematic_tree,
    bound=config["data"]["normalize"] is not None,
    device="cuda",
    verbose=1,
)

# Balance disentanglement losses
config = ssumo.train.losses.balance_disentangle(config, dataset)

optimizer, scheduler = ssumo.train.get_optimizer_and_lr_scheduler(
    model,
    config["train"]["optimizer"],
    config["train"]["lr_scheduler"],
    config["train"]["lr"],
)

if "prior" in config["loss"].keys():
    beta_scheduler = ssumo.train.get_beta_schedule(
        config["loss"]["prior"],
        config["train"]["beta_anneal"],
    )

loss_dict_keys = ["total"] + list(config["loss"].keys())
loss_dict = {k: [] for k in loss_dict_keys}
for epoch in tqdm.trange(
    config["model"]["start_epoch"] + 1, config["train"]["num_epochs"] + 1
):
    if beta_scheduler is not None:
        config["loss"]["prior"] = beta_scheduler.get(epoch)
        print("Beta schedule: {:.3f}".format(config["loss"]["prior"]))

    if "mcmi" in str(config["disentangle"]["method"]):
        print("Running Monte-Carlo mutual information optimization for disentanglement")
        train_func = functools.partial(
            ssumo.train.train_epoch_mcmi,
            var_mode=config["disentangle"]["var_mode"],
            gamma=config["disentangle"]["gamma"],
            bandwidth=config["disentangle"]["bandwidth"],
        )
    else:
        train_func = functools.partial(ssumo.train.train_epoch)
    epoch_loss = train_func(
        model,
        optimizer,
        scheduler,
        loader,
        "cuda",
        config["loss"],
        epoch,
        mode="train",
        disentangle_keys=config["disentangle"]["features"],
    )
    loss_dict = {k: v + [epoch_loss[k]] for k, v in loss_dict.items()}

    if epoch % 10 == 0:
        print("Saving model to folder: {}".format(config["out_path"]))
        torch.save(
            {k: v.cpu() for k, v in model.state_dict().items()},
            "{}/weights/epoch_{}.pth".format(config["out_path"], epoch),
        )

        pickle.dump(
            loss_dict,
            open("{}/losses/loss_dict.p".format(config["out_path"]), "wb"),
        )

        ssumo.plot.eval.loss(
            loss_dict, config["out_path"], config["disentangle"]["features"]
        )
