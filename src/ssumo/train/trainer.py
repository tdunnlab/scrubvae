import torch
from ssumo.train.losses import get_batch_loss
from ssumo.train.mutual_inf import MutInfoEstimator
from ssumo.model.disentangle import MovingAvgLeastSquares
import torch.optim as optim


class CyclicalBetaAnnealing(torch.nn.Module):
    def __init__(self, beta_max=1, len_cycle=100, R=0.5):
        self.beta_max = beta_max
        self.len_cycle = len_cycle
        self.R = R
        self.len_increasing = int(len_cycle * R)

    def get(self, epoch):
        remainder = (epoch-1) % self.len_cycle
        if remainder >= self.len_increasing:
            beta = self.beta_max
        else:
            beta = self.beta_max*remainder/self.len_increasing

        return beta

    

def get_beta_schedule(schedule, beta):
    if schedule == "cyclical":
        print("Initializing cyclical beta annealing")
        beta_scheduler = CyclicalBetaAnnealing(beta_max=beta)
        # cycle_len = n_epochs // M
        # beta_increase = torch.linspace(0, beta ** (1 / 4), int(cycle_len * R)) ** 4
        # beta_plateau = torch.ones(cycle_len - len(beta_increase)) * beta

        # beta_schedule = torch.cat([beta_increase, beta_plateau]).repeat(M)

        # if len(beta_schedule) < n_epochs:
        #     beta_schedule = torch.cat(
        #         [beta_schedule, torch.ones(n_epochs - len(beta_schedule)) * beta]
        #     )
    elif schedule is None:
        print("No beta annealing selected")
        beta_scheduler = None

    return beta_scheduler


def predict_batch(model, data, disentangle_keys=None):

    data_i = {
        k: v
        for k, v in data.items()
        if (k in disentangle_keys) or (k in ["x6d", "root"])
    }

    return model(data_i)

def get_optimizer_and_lr_scheduler(model, optimization="adamw", lr_schedule="cawr", lr=1e-7):
    if optimization == "adam":
        print("Initializing Adam optimizer ...")
        optimizer = optim.Adam(model.parameters(), lr=lr)
    elif optimization == "adamw":
        print("Initializing AdamW optimizer ...")
        optimizer = optim.AdamW(model.parameters(), lr=lr)
    elif optimization == "sgd":
        print("Initializing SGD optimizer ...")
        optimizer = optim.SGD(
            model.parameters(), lr=lr, momentum=0.2, nesterov=True
        )
    else:
        raise ValueError("No valid optimizer selected")

    if lr_schedule == "cawr":
        print("Initializing cosine annealing w/warm restarts learning rate scheduler")
        scheduler = optim.lr_scheduler.CosineAnnealingWarmRestarts(optimizer, T_0=50)
    elif lr_schedule is None:
        print("No learning rate scheduler selected")
        scheduler = None

    return optimizer, scheduler

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

            batch_loss = get_batch_loss(data, data_o, loss_config, )

            # if len(model.disentangle.keys())>0:
            #     if isinstance(model.disentangle.values()[0], MovingAvgLeastSquares):
            #         for k,v in model.disentangle.items():
            #             batch_loss += 

            if mode == "train":
                batch_loss["total"].backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1e5)
                optimizer.step()
                if scheduler is not None:
                    scheduler.step(epoch + batch_idx / len(loader))

            epoch_loss = {k: v + batch_loss[k].detach() for k, v in epoch_loss.items()}

        for k, v in epoch_loss.items():
            epoch_loss[k] = v.item() / len(loader.dataset)
            print(
                "====> Epoch: {} Average {} loss: {:.4f}".format(
                    epoch, k, epoch_loss[k]
                )
            )

    return epoch_loss


def train_epoch_mcmi(
    model,
    optimizer,
    scheduler,
    loader,
    device,
    loss_config,
    epoch,
    disentangle_keys,
    bandwidth=1,
    var_mode="sphere",
    gamma=1,
    mode="train",
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

            variables = torch.cat([data[k] for k in model.disentangle_keys], dim=-1)
            batch_loss = get_batch_loss(data, data_o, loss_config)
            if batch_idx > 0:
                batch_loss["mcmi"] = mi_estimator(data_o["mu"], variables)
                # batch_loss["total"] += batch_loss["mcmi"]*1000
                batch_loss["total"] += loss_config["mcmi"]*batch_loss["mcmi"]
            else:
                batch_loss["mcmi"] = batch_loss["total"]

            if mode == "train":
                batch_loss["total"].backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1e5)

                optimizer.step()
                if scheduler is not None:
                    scheduler.step(epoch + batch_idx / len(loader))

            if var_mode == "diagonal":
                L_sample = data_o["L"].detach().clone()
                var_sample = L_sample.diagonal(dim1=-2, dim2=-1) ** 2 + bandwidth
            else:
                var_sample = bandwidth
            
            mi_estimator = MutInfoEstimator(
                data_o["mu"].detach().clone(),
                variables.clone(),
                var_sample,
                gamma=gamma,
                var_mode=var_mode,
                device=device,
            )
            epoch_loss = {k: v + batch_loss[k].detach() for k, v in epoch_loss.items()}

        for k, v in epoch_loss.items():
            epoch_loss[k] = v.item() / len(loader)
            print(
                "====> Epoch: {} Average {} loss: {:.4f}".format(
                    epoch, k, epoch_loss[k]
                )
            )

    return epoch_loss
