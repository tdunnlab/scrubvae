import torch
from dappy import read
from ssumo.train.losses import get_batch_loss, balance_disentangle
from ssumo.train.mutual_inf import MutInfoEstimator
from ssumo.model.disentangle import MovingAvgLeastSquares, QuadraticDiscriminantFilter
from ssumo.plot.eval import loss as plt_loss
from ssumo.get.data import projected_2D_kinematics
import torch.optim as optim
import tqdm
import pickle
import functools
import time
import random
from math import pi, sin, cos


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
    disentangle_config,
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
            disentangle_config,
        )

        if mode == "train":
            batch_loss["total"].backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1e7)
            optimizer.step()
            if scheduler is not None:
                scheduler.step(epoch + batch_idx / len(loader))

            if bool(model.disentangle):
                for method in model.disentangle.keys():
                    if method in ["moving_avg_lsq", "moving_avg", "qda"]:
                        for k in model.disentangle[method].keys():
                            model.disentangle[method][k].update(
                                data_o["mu"].detach().clone(), data[k].detach().clone()
                            )

        epoch_loss = {k: v + batch_loss[k].detach() for k, v in epoch_loss.items()}

    return epoch_loss


@epoch_wrapper
def train_epoch_2D_view(
    model,
    optimizer,
    scheduler,
    loader,
    device,
    loss_config,
    disentangle_config,
    epoch,
    config,
    skeleton_config,
    mode="train",
):
    epoch_loss = {k: 0 for k in ["total"] + list(loss_config.keys())}
    for batch_idx, data in enumerate(loader):
        if mode == "train":
            for param in model.parameters():
                param.grad = None
        axis = random.random() * pi / 2
        axis = [0, -cos(axis), -sin(axis)]
        data["view_axis"] = torch.tensor(axis)[None, :].repeat(
            (len(data["raw_pose"]), 1)
        )
        data = {k: v.to(device) for k, v in data.items()}
        data = projected_2D_kinematics(
            data,
            axis,
            config,
            skeleton_config,
            device=device,
        )
        data = {k: v.to(device) for k, v in data.items()}
        data_o = predict_batch(model, data, model.disentangle_keys)

        batch_loss = get_batch_loss(
            model,
            data,
            data_o,
            loss_config,
            disentangle_config,
        )

        if mode == "train":
            batch_loss["total"].backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1e7)
            optimizer.step()
            if scheduler is not None:
                scheduler.step(epoch + batch_idx / len(loader))

            if bool(model.disentangle):
                for method in model.disentangle.keys():
                    if method in ["moving_avg_lsq", "moving_avg", "qda"]:
                        for k in model.disentangle[method].keys():
                            model.disentangle[method][k].update(
                                data_o["mu"].detach().clone(), data[k].detach().clone()
                            )

        epoch_loss = {k: v + batch_loss[k].detach() for k, v in epoch_loss.items()}

    return epoch_loss


@epoch_wrapper
def train_epoch_mcmi_2D_view(
    model,
    optimizer,
    scheduler,
    loader,
    device,
    loss_config,
    disentangle_config,
    epoch,
    bandwidth=1,
    var_mode="sphere",
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
        batch_loss = get_batch_loss(
            model, data, data_o, loss_config, disentangle_config
        )
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

        updated_data_o = model.encode(data)
        if var_mode == "diagonal":
            L_sample = updated_data_o["L"].detach().clone()
            var_sample = L_sample.diagonal(dim1=-2, dim2=-1) ** 2 + bandwidth
        else:
            var_sample = bandwidth

        mi_estimator = MutInfoEstimator(
            updated_data_o["mu"].detach().clone(),
            variables.clone(),
            var_sample,
            bandwidth=bandwidth,
            var_mode=var_mode,
            device=device,
        )
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
    disentangle_config,
    epoch,
    bandwidth=1,
    var_mode="sphere",
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
        batch_loss = get_batch_loss(
            model, data, data_o, loss_config, disentangle_config
        )
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

        updated_data_o = model.encode(data)
        if var_mode == "diagonal":
            L_sample = updated_data_o["L"].detach().clone()
            var_sample = L_sample.diagonal(dim1=-2, dim2=-1) ** 2 + bandwidth
        else:
            var_sample = bandwidth

        mi_estimator = MutInfoEstimator(
            updated_data_o["mu"].detach().clone(),
            variables.clone(),
            var_sample,
            bandwidth=bandwidth,
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
                bandwidth=config["disentangle"]["bandwidth"],
            )
            if config["data"].get("is_2D") == True:
                train_func = functools.partial(
                    train_epoch_mcmi_2D_view,
                    var_mode=config["disentangle"]["var_mode"],
                    bandwidth=config["disentangle"]["bandwidth"],
                    data_config=config["data"],
                )
        else:
            train_func = functools.partial(train_epoch)
            skeleton_config = read.config(config["data"]["skeleton_path"])
            if config["data"].get("is_2D") == True:
                train_func = functools.partial(
                    train_epoch_2D_view,
                    config=config,
                    skeleton_config=skeleton_config,
                )

        starttime = time.time()
        epoch_loss = train_func(
            model=model,
            optimizer=optimizer,
            scheduler=scheduler,
            loader=loader,
            device="cuda",
            loss_config=config["loss"],
            disentangle_config=config["disentangle"],
            epoch=epoch,
            mode="train",
        )

        if "grad_reversal" in model.disentangle.keys():
            for key in model.disentangle["grad_reversal"].keys():
                model.disentangle["grad_reversal"][key].reset_parameters()

        epoch_loss["time"] = time.time() - starttime
        epoch_loss["epoch"] = epoch
        loss_dict = {k: v + [epoch_loss[k]] for k, v in loss_dict.items()}

        if epoch % 5 == 0:
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
