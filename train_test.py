from ssumo.data.dataset import inv_normalize_root
from ssumo import model, train
import ssumo
from torch.utils.data import DataLoader
import torch

torch.autograd.set_detect_anomaly(True)
import torch.optim as optim
import matplotlib.pyplot as plt
import numpy as np
import tqdm
from ssumo.plot.constants import PALETTE_DICT
from ssumo.parameters import read
import pickle
import sys

### Set/Load Parameters
base_path = "/mnt/ceph/users/jwu10/results/vae/heading/"
analysis_key = sys.argv[1]
print(analysis_key)
config = read.config(base_path + analysis_key + "/model_config.yaml")

### Load Dataset
dataset = ssumo.data.get_mouse(
    data_config=config["data"],
    window=config["model"]["window"],
    train=True,
    data_keys=["x6d", "root", "offsets"] + config["disentangle"]["features"],
)

loader = DataLoader(
    dataset=dataset, batch_size=config["train"]["batch_size"], shuffle=True
)
vae, device = ssumo.model.get(
    config["model"],
    config["disentangle"],
    dataset.n_keypts,
    config["data"]["direction_process"],
    verbose=1,
)
optimizer = optim.Adam(vae.parameters(), lr=0.0001)
vae.train()
# else:
#     vae, speed_decoder, device = utils.init_model(
#         config, dataset.n_keypts, config["invariant"]
#     )
#     optimizer = optim.Adam(
#         params=list(vae.parameters()) + list(speed_decoder.parameters()), lr=0.0001
#     )
#     speed_decoder.train()

# vae.train()

beta_schedule = train.trainer.get_beta_schedule(
    config["loss"]["prior"],
    config["train"]["num_epochs"] - config["model"]["load_epoch"],
    config["train"]["beta_anneal"],
)

loss_dict_keys = ["total"] + list(config["loss"].keys())
loss_dict = {k: [0] for k in loss_dict_keys}
for epoch in tqdm.trange(
    config["model"]["load_epoch"] + 1, config["train"]["num_epochs"] + 1
):
    config["loss"]["prior"] = beta_schedule[epoch - config["model"]["load_epoch"] - 1]
    print("Beta schedule: {}".format(config["loss"]["prior"]))
    for batch_idx, data in enumerate(loader):
        optimizer.zero_grad()
        data = {k: v.to(device) for k, v in data.items()}
        data_o = {
            "kinematic_tree": dataset.kinematic_tree,
            "arena_size": dataset.arena_size.cuda(),
        }
        len_batch = len(data["x6d"])

        # if (config["invariant"] == "speed") or (
        #     config["speed_decoder"] == "avg" or "part"
        # ):
        #     data["speed"] = data["speed"].mean(dim=1)

        #     if len(data["speed"].shape) < 2:
        #         data["speed"] = data["speed"][:, None]

        if config["disentangle"]["method"] == "invariant":
            invariant = torch.cat( [data[key] for key in config["disentangle"]["features"]], axis=-1 )
        else:
            invariant = None

        if config["data"]["arena_size"] is not None:
            x_i = torch.cat(
                (data["x6d"].view(data["x6d"].shape[:2] + (-1,)), data["root"]), axis=-1
            )
            x_o, data_o["mu"], data_o["L"], data_o["disentangle"] = vae( x_i, invariant=invariant )

            data_o["x6d"] = x_o[..., :-3].reshape(data["x6d"].shape)
            data_o["root"] = inv_normalize_root(x_o[..., -3:], data_o["arena_size"])
            data["root"] = inv_normalize_root(data["root"], data_o["arena_size"])

        else:
            data_o["x6d"], data_o["mu"], data_o["L"], data_o["disentangle"] = vae(
                data["x6d"], invariant=invariant
            )

        # if config["disentangle"]["method"] is not None:
        #     for 

        # if config["speed_decoder"] is not None:
        #     if config["gradient_reversal"]:
        #         data_o["speed"], data_o["speed_gr"] = speed_decoder(data_o["mu"])
        #     else:
        #         data_o["speed"] = speed_decoder(data_o["mu"])

        batch_loss = train.losses.get_batch_loss(data, data_o, config["loss"])
        batch_loss["total"].backward()
        optimizer.step()

        # import pdb; pdb.set_trace()
        loss_dict = {
            k: v[:-1] + [v[-1] + batch_loss[k].item()] for k, v in loss_dict.items()
        }

        if batch_idx % 5000 == 0:
            print(
                "Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}".format(
                    epoch,
                    batch_idx * len_batch,
                    len(loader.dataset),
                    100.0 * batch_idx / len(loader),
                    batch_loss["total"].item() / len_batch,
                )
            )

    for k, v in loss_dict.items():
        loss_dict[k][-1] = v[-1] / len(loader.dataset)
        print("====> Epoch: {} Average {} loss: {:.4f}".format(epoch, k, v[-1]))
        loss_dict[k] += [0]

    if epoch % 10 == 0:
        print("Saving model to folder: {}".format(config["out_path"]))
        torch.save(
            vae.state_dict(),
            "{}/weights/epoch_{}.pth".format(config["out_path"], epoch),
        )

        pickle.dump(
            loss_dict,
            open("{}/losses/loss_dict.pth".format(config["out_path"]), "wb"),
        )

        # if config["speed_decoder"]:
        #     torch.save(
        #         speed_decoder.state_dict(),
        #         "{}/weights/{}_spd_epoch_{}.pth".format(
        #             config["out_path"], config["speed_decoder"], epoch
        #         ),
        #     )

        f = plt.figure(figsize=(15, 10))
        for i, (k, v) in enumerate(loss_dict.items()):
            plt.plot(np.arange(1, len(v)), v[:-1], label=k, c=PALETTE_DICT[i])

        plt.yscale("log")
        plt.xlabel("Epoch")
        plt.ylabel("Log Loss")
        plt.legend()
        plt.savefig("{}/losses/loss_epoch.png".format(config["out_path"]))
        plt.close()
