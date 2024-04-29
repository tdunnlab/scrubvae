import torch
from ssumo.train.losses import get_batch_loss, balance_disentangle
from ssumo.train.mutual_inf import MutInfoEstimator
from ssumo.model.disentangle import MovingAvgLeastSquares, QuadraticDiscriminantFilter
from ssumo.plot.eval import loss as plt_loss
import torch.optim as optim
import tqdm
import pickle
import functools
import time


class CyclicalBetaAnnealing(torch.nn.Module):
    def __init__(self, beta_max=1, len_cycle=100, R=0.5):
        self.beta_max = beta_max
        self.len_cycle = len_cycle
        self.R = R
        self.len_increasing = int(len_cycle * R)

    def get(self, epoch):
        remainder = (epoch - 1) % self.len_cycle
        if remainder >= self.len_increasing:
            beta = self.beta_max
        else:
            beta = self.beta_max * remainder / self.len_increasing

        return beta

def get_beta_schedule(schedule, beta):
    if schedule == "cyclical":
        print("Initializing cyclical beta annealing")
        beta_scheduler = CyclicalBetaAnnealing(beta_max=beta)
    else:
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


def get_optimizer_and_lr_scheduler(
    model, optimization="adamw", lr_schedule="cawr", lr=1e-7
):
    if optimization == "adam":
        print("Initializing Adam optimizer ...")
        optimizer = optim.Adam(model.parameters(), lr=lr)
    elif optimization == "adamw":
        print("Initializing AdamW optimizer ...")
        optimizer = optim.AdamW(model.parameters(), lr=lr)
    elif optimization == "sgd":
        print("Initializing SGD optimizer ...")
        optimizer = optim.SGD(model.parameters(), lr=lr, momentum=0.2, nesterov=True)
    else:
        raise ValueError("No valid optimizer selected")

    if lr_schedule == "cawr":
        print("Initializing cosine annealing w/warm restarts learning rate scheduler")
        scheduler = optim.lr_scheduler.CosineAnnealingWarmRestarts(optimizer, T_0=50)
    elif lr_schedule is None:
        print("No learning rate scheduler selected")
        scheduler = None

    return optimizer, scheduler


def epoch_wrapper(func):
    @functools.wraps(func)
    def wrapper(
        model,
        optimizer,
        scheduler,
        loader,
        device,
        loss_config,
        epoch,
        mode="train",
        **kwargs,
    ):
        if mode == "train":
            model.train()
            grad_env = torch.enable_grad
        elif ("test" or "encode" or "decode") in mode:
            model.eval()
            grad_env = torch.no_grad
        else:
            raise ValueError("This mode is not recognized.")

        with grad_env():
            epoch_loss = func(
                model=model,
                optimizer=optimizer,
                scheduler=scheduler,
                loader=loader,
                device=device,
                loss_config=loss_config,
                epoch=epoch,
                mode=mode,
                **kwargs,
            )

        for k, v in epoch_loss.items():
            epoch_loss[k] = v.item() / len(loader)
            print(
                "====> Epoch: {} Average {} loss: {:.4f}".format(
                    epoch, k, epoch_loss[k]
                )
            )
        return epoch_loss

    return wrapper


@epoch_wrapper
def train_epoch(
    model,
    optimizer,
    scheduler,
    loader,
    device,
    loss_config,
    epoch,
    mode="train",
):
    epoch_loss = {k: 0 for k in ["total"] + list(loss_config.keys())}
    for batch_idx, data in enumerate(loader):
        if mode == "train":
            for param in model.parameters():
                param.grad = None
        data = {k: v.to(device) for k, v in data.items()}
        data_o = predict_batch(model, data, model.disentangle_keys)

        batch_loss = get_batch_loss(
            model,
            data,
            data_o,
            loss_config,
        )

        if mode == "train":
            batch_loss["total"].backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1e7)
            optimizer.step()
            if scheduler is not None:
                scheduler.step(epoch + batch_idx / len(loader))

            if bool(model.disentangle):
                for k, v in model.disentangle.items():
                    if isinstance(v, MovingAvgLeastSquares) or isinstance(
                        v, QuadraticDiscriminantFilter
                    ):
                        model.disentangle[k].update(data_o["mu"], data[k])

        epoch_loss = {k: v + batch_loss[k].detach() for k, v in epoch_loss.items()}

    return epoch_loss


@epoch_wrapper
def train_epoch_mcmi(
    model,
    optimizer,
    scheduler,
    loader,
    device,
    loss_config,
    epoch,
    bandwidth=1,
    var_mode="sphere",
    gamma=1,
    mode="train",
):
    epoch_loss = {k: 0 for k in ["total"] + list(loss_config.keys())}
    for batch_idx, data in enumerate(loader):
        if mode == "train":
            for param in model.parameters():
                param.grad = None

        data = {k: v.to(device) for k, v in data.items()}
        data_o = predict_batch(model, data, model.disentangle_keys)

        variables = torch.cat([data[k] for k in model.disentangle_keys], dim=-1)
        batch_loss = get_batch_loss(model, data, data_o, loss_config)
        if batch_idx > 0:
            batch_loss["mcmi"] = mi_estimator(data_o["mu"], variables)
            batch_loss["total"] += loss_config["mcmi"] * batch_loss["mcmi"]
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

    return epoch_loss


def train(config, model, loader):
    torch.autograd.set_detect_anomaly(True)
    torch.backends.cudnn.benchmark = True
    config = balance_disentangle(config, loader.dataset)

    optimizer, scheduler = get_optimizer_and_lr_scheduler(
        model,
        config["train"]["optimizer"],
        config["train"]["lr_schedule"],
        config["train"]["lr"],
    )

    if "prior" in config["loss"].keys():
        beta_scheduler = get_beta_schedule(
            config["loss"]["prior"],
            config["train"]["beta_anneal"],
        )
    else:
        beta_scheduler = None

    if config["model"]["load_model"] == config["out_path"]:
        try:
            loss_dict = pickle.load(
                open(
                    "{}/losses/loss_dict.p".format(config["model"]["load_model"]), "rb"
                )
            )
        except:
            loss_dict = pickle.load(
                open(
                    "{}/losses/loss_dict_Train.p".format(config["model"]["load_model"]),
                    "rb",
                )
            )
    else:
        loss_dict_keys = ["total", "time", "epoch"] + list(config["loss"].keys())
        loss_dict = {k: [] for k in loss_dict_keys}

    for epoch in tqdm.trange(
        config["model"]["start_epoch"] + 1, config["train"]["num_epochs"] + 1
    ):
        if beta_scheduler is not None:
            config["loss"]["prior"] = beta_scheduler.get(epoch)
            print("Beta schedule: {:.3f}".format(config["loss"]["prior"]))

        if "mcmi" in str(config["disentangle"]["method"]):
            print(
                "Running Monte-Carlo mutual information optimization for disentanglement"
            )
            train_func = functools.partial(
                train_epoch_mcmi,
                var_mode=config["disentangle"]["var_mode"],
                gamma=config["disentangle"]["gamma"],
                bandwidth=config["disentangle"]["bandwidth"],
            )
        else:
            train_func = functools.partial(train_epoch)

        starttime = time.time()
        epoch_loss = train_func(
            model=model,
            optimizer=optimizer,
            scheduler=scheduler,
            loader=loader,
            device="cuda",
            loss_config=config["loss"],
            epoch=epoch,
            mode="train",
        )
        epoch_loss["time"] = time.time() - starttime
        epoch_loss["epoch"] = epoch
        loss_dict = {k: v + [epoch_loss[k]] for k, v in loss_dict.items()}

        if epoch % 2 == 0:
            print("Saving model to folder: {}".format(config["out_path"]))
            torch.save(
                {k: v.cpu() for k, v in model.state_dict().items()},
                "{}/weights/epoch_{}.pth".format(config["out_path"], epoch),
            )

            pickle.dump(
                loss_dict,
                open("{}/losses/loss_dict_Train.p".format(config["out_path"]), "wb"),
            )

            plt_loss(loss_dict, config["out_path"], config["disentangle"]["features"])

    return model
