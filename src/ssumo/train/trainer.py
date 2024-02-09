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
        beta_schedule = torch.zeros(n_epochs) + beta

    return beta_schedule.type(torch.float32).detach().numpy()


def predict_batch(model, data, disentangle_keys=None):

    data_i = {
        k: v
        for k, v in data.items()
        if (k in disentangle_keys) or (k in ["x6d", "root"])
    }

    return model(data_i)


def train_epoch(
    model,
    optimizer,
    scheduler,
    loader,
    device,
    loss_config,
    epoch,
    mode="train",
    disentangle_keys=None,
):
    if mode == "train":
        model.train()
        grad_env = torch.enable_grad
    elif ("test" or "encode" or "decode") in mode:
        model.eval()
        grad_env = torch.no_grad
    else:
        raise ValueError("This mode is not recognized.")
    epoch_loss = {k: 0 for k in ["total"] + list(loss_config.keys())}
    with grad_env():
        for batch_idx, data in enumerate(loader):
            if mode == "train":
                for param in model.parameters():
                    param.grad = None

            data = {k: v.to(device) for k, v in data.items()}
            data["kinematic_tree"] = model.kinematic_tree
            data_o = predict_batch(model, data, disentangle_keys)

            batch_loss = get_batch_loss(data, data_o, loss_config)

            if mode == "train":
                batch_loss["total"].backward()
                # total_norm = 0
                # for p in model.parameters():
                #     param_norm = p.grad.data.norm(2)
                #     total_norm += param_norm.item() ** 2
                # total_norm = total_norm ** (1. / 2)
                # print(total_norm)

                # torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1e5)
                optimizer.step()
                scheduler.step(epoch + batch_idx / len(loader))
            epoch_loss = {k: v + batch_loss[k].detach() for k, v in epoch_loss.items()}

            # if batch_idx % 500 == 0:
            #     len_batch = len(data["x6d"])
            #     print(
            #         "{} Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}".format(
            #             mode.title(),
            #             epoch,
            #             batch_idx * len_batch,
            #             len(loader.dataset),
            #             100.0 * batch_idx / len(loader),
            #             batch_loss["total"].item() / len_batch,
            #         )
            #     )

        for k, v in epoch_loss.items():
            epoch_loss[k] = v.item() / len(loader.dataset)
            print(
                "====> Epoch: {} Average {} loss: {:.4f}".format(
                    epoch, k, epoch_loss[k]
                )
            )

    return epoch_loss
