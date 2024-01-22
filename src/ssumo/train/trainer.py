import torch
from ssumo.train.losses import get_batch_loss

def get_beta_schedule(beta, n_epochs, beta_anneal=False, M=4, R=0.75):
    if beta_anneal:
        print("Cyclical beta anneal")
        cycle_len = n_epochs // M
        beta_increase = torch.linspace(0, beta ** (1 / 4), int(cycle_len * R)) ** 4
        beta_plateau = torch.ones(cycle_len - len(beta_increase)) * beta

        beta_schedule = torch.cat([beta_increase, beta_plateau]).repeat(M)

        if len(beta_schedule) < n_epochs:
            beta_schedule = torch.cat(
                [beta_schedule, torch.ones(n_epochs - len(beta_schedule)) * beta]
            )
    else:
        print("No beta anneal")
        beta_schedule = torch.ones(n_epochs) * beta

    return beta_schedule


def predict_batch(vae, data, disentangle_keys=None):
    # data_o = {}

    # if vae.invariant_dim > 0:
    #     invariant = torch.cat([data[key] for key in disentangle_keys], axis=-1)
    # else:
    #     invariant = None

    # if "arena_size" in data.keys():
    #     # x_i = torch.cat(
    #     #     (data["x6d"].view(data["x6d"].shape[:2] + (-1,)), data["root"]), axis=-1
    #     # )
    #     x_o, data_o["mu"], data_o["L"], data_o["disentangle"] = vae(
    #         x_i, invariant=invariant
    #     )

    #     data_o["x6d"] = x_o[..., :-3].reshape(data["x6d"].shape)
    #     import pdb; pdb.set_trace()
    #     data_o["root"] = inv_normalize_root(x_o[..., -3:], data["arena_size"])
    #     data["root"] = inv_normalize_root(data["root"], data["arena_size"])

    # else:
    #     data_o["x6d"], data_o["mu"], data_o["L"], data_o["disentangle"] = vae(
    #         data["x6d"], invariant=invariant
    #     )
    data_i = {k:v for k,v in data.items() if (k in disentangle_keys) or (k in ["x6d","root"])}

    return vae(data_i)


def train_epoch(vae, optimizer, loader, device, loss_config, epoch, mode="train", disentangle_keys=None):
    if mode == "train":
        vae.train()
        grad_env = torch.enable_grad
    elif ("test" or "encode" or "decode") in mode:
        vae.eval()
        grad_env = torch.no_grad
    else:
        raise ValueError("This mode is not recognized.")
    epoch_loss = {k: 0 for k in ["total"] + list(loss_config.keys())}
    with grad_env():
        for batch_idx, data in enumerate(loader):
            if mode == "train":
                optimizer.zero_grad()

            data = {k: v.to(device) for k, v in data.items()}
            data["kinematic_tree"] = vae.kinematic_tree
            len_batch = len(data["x6d"])
            data_o = predict_batch(vae, data, disentangle_keys)

            batch_loss = get_batch_loss(data, data_o, loss_config)

            if mode == "train":
                batch_loss["total"].backward()
                optimizer.step()
            epoch_loss = { k: v + batch_loss[k].item() for k, v in epoch_loss.items() }

            if batch_idx % 500 == 0:
                print(
                    "{} Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}".format(
                        mode.title(),
                        epoch,
                        batch_idx * len_batch,
                        len(loader.dataset),
                        100.0 * batch_idx / len(loader),
                        batch_loss["total"].item() / len_batch,
                    )
                )

        for k, v in epoch_loss.items():
            epoch_loss[k] = v / len(loader.dataset)
            print("====> Epoch: {} Average {} loss: {:.4f}".format(epoch, k, epoch_loss[k]))

    return epoch_loss