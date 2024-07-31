import torch
from ssumo.train.losses import get_batch_loss, balance_disentangle
from ssumo.train.mutual_inf import MutInfoEstimator
from ssumo.model.disentangle import MovingAvgLeastSquares, QuadraticDiscriminantFilter
from ssumo.plot.eval import loss as plt_loss
from ssumo.eval import generative_restrictiveness
from ssumo.eval import cluster
import torch.optim as optim
import tqdm
import pickle
import functools
import time
import wandb
from ssumo.eval.metrics import (
    linear_rand_cv,
    mlp_rand_cv,
    log_class_rand_cv,
    qda_rand_cv,
)
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import r2_score
import numpy as np


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


def train_test_epoch(
    config,
    model,
    loader,
    device,
    epoch,
    optimizer=None,
    scheduler=None,
    mode="train",
    get_z=False,
):
    if mode == "train":
        model.train()
        grad_env = torch.enable_grad
    elif mode == "test":
        model.eval()
        grad_env = torch.no_grad
    else:
        raise ValueError("This mode is not recognized.")
    with grad_env():
        z = []
        model.mi_estimator = None
        epoch_metrics = {k: 0 for k in ["total"] + list(config["loss"].keys())}
        for batch_idx, data in enumerate(loader):
            data = {k: v.to(device) for k, v in data.items()}
            data_o = predict_batch(model, data, model.disentangle_keys)

            if get_z:
                z += [data_o["mu"].clone().detach().cpu()]

            batch_loss = get_batch_loss(
                model,
                data,
                data_o,
                config["loss"],
                config["disentangle"],
            )

            # if "mcmi" in config["loss"]:
            #     variables = torch.cat(
            #         [data[k] for k in model.disentangle_keys], dim=-1
            #     )
            #     if batch_idx > 0:
            #         batch_loss["mcmi"] = mi_estimator(data_o["mu"], variables)
            #         batch_loss["total"] += (
            #             config["loss"]["mcmi"] * batch_loss["mcmi"]
            #         )
            #     else:
            #         batch_loss["mcmi"] = 0

            if mode == "train":
                for param in model.parameters():
                    param.grad = None

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
                                    data_o["mu"].detach().clone(),
                                    data[k].detach().clone(),
                                )

            epoch_metrics = {
                k: v + batch_loss[k].detach() for k, v in epoch_metrics.items()
            }

            if "mcmi" in config["loss"]:
                updated_data_o = model.encode(data)
                variables = torch.cat([data[k] for k in model.disentangle_keys], dim=-1)

                # if config["disentangle"]["var_mode"] == "diagonal":
                #     L_sample = updated_data_o["L"].detach().clone()
                #     var_sample = (
                #         L_sample.diagonal(dim1=-2, dim2=-1) ** 2
                #         + config["disentangle"]["bandwidth"]
                #     )
                # else:
                #     var_sample = config["disentangle"]["bandwidth"]

                model.mi_estimator = MutInfoEstimator(
                    x_s=updated_data_o["mu"].detach().clone(),
                    y_s=variables.detach().clone(),
                    bandwidth=config["disentangle"]["bandwidth"],
                    var_mode=config["disentangle"]["var_mode"],
                    model_var=(
                        updated_data_o["L"].detach().clone()
                        if "L" in updated_data_o.keys()
                        else None
                    ),
                    device=device,
                )

    for k, v in epoch_metrics.items():
        epoch_metrics[k] = v.item() / len(loader)
        print(
            "====> Epoch: {} Average {} loss: {:.4f}".format(epoch, k, epoch_metrics[k])
        )

    if get_z:
        return epoch_metrics, torch.cat(z, dim=0)
    else:
        return epoch_metrics, 0


# def test_epoch(config, model, loader, device="cuda", epoch=0):
#     epoch_metrics, z = train_test_epoch(
#         config=config,
#         model=model,
#         loader=loader,
#         device=device,
#         epoch=epoch,
#         mode="test",
#         get_z=True,
#     )
#     # epoch_metrics = {k: v + batch_loss[k].detach() for k, v in epoch_metrics.items()}
#     return epoch_metrics, z


def test_epoch(config, model, loader, device="cuda", epoch=0):
    loader.dataset.data["avg_speed_3d_rand"] = loader.dataset[:]["avg_speed_3d"][
        torch.randperm(
            len(loader.dataset), generator=torch.Generator().manual_seed(100)
        )
    ]
    model.eval()
    with torch.no_grad():
        z = []

        model.mi_estimator = None
        epoch_metrics = {k: 0 for k in ["total"] + list(config["loss"].keys())}
        gen_res = {
            k1: {k2: [] for k2 in ["pred", "target"]}
            for k1 in ["heading", "avg_speed_3d"]
        }
        for batch_idx, data in enumerate(loader):
            data = {k: v.to(device) for k, v in data.items()}
            data_o = predict_batch(model, data, model.disentangle_keys)

            z = [data_o["mu"].clone().detach().cpu()]

            batch_metrics = get_batch_loss(
                model,
                data,
                data_o,
                config["loss"],
                config["disentangle"],
            )

            for key in gen_res.keys():
                key_pred, key_target = generative_restrictiveness(
                    model, data_o["mu"], data, key, loader.dataset.kinematic_tree
                )
                if "speed" in key:
                    norm_params = {
                        k: v.to(key_pred.device)
                        for k, v in loader.dataset.norm_params[key].items()
                    }
                    if "mean" in norm_params.keys():
                        key_pred -= norm_params["mean"]
                        key_pred /= norm_params["std"]
                    elif "min" in norm_params.keys():
                        key_pred -= norm_params["min"]
                        key_pred = 2 * key_pred / norm_params["max"] - 1

                gen_res[key]["pred"] += [key_pred.detach().cpu()]
                gen_res[key]["target"] += [key_target.detach().cpu()]

            epoch_metrics = {
                k: v + batch_metrics[k].detach() for k, v in epoch_metrics.items()
            }

            if "mcmi" in config["loss"]:
                updated_data_o = model.encode(data)
                variables = torch.cat([data[k] for k in model.disentangle_keys], dim=-1)

                model.mi_estimator = MutInfoEstimator(
                    x_s=updated_data_o["mu"].detach().clone(),
                    y_s=variables.detach().clone(),
                    bandwidth=config["disentangle"]["bandwidth"],
                    var_mode=config["disentangle"]["var_mode"],
                    model_var=(
                        updated_data_o["L"].detach().clone()
                        if "L" in updated_data_o.keys()
                        else None
                    ),
                    device=device,
                )

    for k, v in epoch_metrics.items():
        epoch_metrics[k] = v.item() / len(loader)
        print(
            "====> Epoch: {} Average {} loss: {:.4f}".format(epoch, k, epoch_metrics[k])
        )

    for key in gen_res.keys():
        epoch_metrics["r2_gen_restrict_{}".format(key)] = r2_score(
            torch.cat(gen_res[key]["target"], dim=0),
            torch.cat(gen_res[key]["pred"], dim=0),
        )

    return epoch_metrics, torch.cat(z, dim=0)


def train_epoch(config, model, loader, optimizer, scheduler, device="cuda", epoch=0):
    epoch_metrics = train_test_epoch(
        config=config,
        model=model,
        loader=loader,
        optimizer=optimizer,
        scheduler=scheduler,
        device=device,
        epoch=epoch,
        mode="train",
        get_z=epoch % 5 == 0,
    )

    return epoch_metrics


# @epoch_wrapper
# def train_epoch_mcmi(
#     model,
#     optimizer,
#     scheduler,
#     loader,
#     device,
#     loss_config,
#     disentangle_config,
#     epoch,
#     bandwidth=1,
#     var_mode="sphere",
#     mode="train",
# ):
#     epoch_loss = {k: 0 for k in ["total"] + list(loss_config.keys())}
#     for batch_idx, data in enumerate(loader):
#         if mode == "train":
#             for param in model.parameters():
#                 param.grad = None

#         data = {k: v.to(device) for k, v in data.items()}
#         data_o = predict_batch(model, data, model.disentangle_keys)

#         batch_loss = get_batch_loss(
#             model, data, data_o, loss_config, disentangle_config
#         )

#         if mode == "train":
#             batch_loss["total"].backward()
#             torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1e5)

#             optimizer.step()
#             if scheduler is not None:
#                 scheduler.step(epoch + batch_idx / len(loader))

#         updated_data_o = model.encode(data)
#         if var_mode == "diagonal":
#             L_sample = updated_data_o["L"].detach().clone()
#             var_sample = L_sample.diagonal(dim1=-2, dim2=-1) ** 2 + bandwidth
#         else:
#             var_sample = bandwidth

#         mi_estimator = MutInfoEstimator(
#             updated_data_o["mu"].detach().clone(),
#             variables.clone(),
#             var_sample,
#             bandwidth=bandwidth,
#             var_mode=var_mode,
#             device=device,
#         )

#         epoch_loss = {k: v + batch_loss[k].detach() for k, v in epoch_loss.items()}

#     return epoch_loss


def train(config, model, train_loader, test_loader, run=None):
    torch.autograd.set_detect_anomaly(True)
    torch.backends.cudnn.benchmark = True
    config = balance_disentangle(config, train_loader.dataset)

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

    # if config["model"]["load_model"] == config["out_path"]:
    #     try:
    #         loss_dict = pickle.load(
    #             open(
    #                 "{}/losses/loss_dict.p".format(config["model"]["load_model"]), "rb"
    #             )
    #         )
    #     except:
    #         loss_dict = pickle.load(
    #             open(
    #                 "{}/losses/loss_dict_Train.p".format(config["model"]["load_model"]),
    #                 "rb",
    #             )
    #         )
    # else:
    #     loss_dict_keys = ["total", "time", "epoch"] + list(config["loss"].keys())
    #     loss_dict = {k: [] for k in loss_dict_keys}

    for epoch in tqdm.trange(
        config["model"]["start_epoch"] + 1, config["train"]["num_epochs"] + 1
    ):
        if beta_scheduler is not None:
            config["loss"]["prior"] = beta_scheduler.get(epoch)
            print("Beta schedule: {:.3f}".format(config["loss"]["prior"]))

        # if "mcmi" in str(config["disentangle"]["method"]):
        #     print(
        #         "Running Monte-Carlo mutual information optimization for disentanglement"
        #     )
        #     train_func = functools.partial(
        #         train_epoch_mcmi,
        #         var_mode=config["disentangle"]["var_mode"],
        #         bandwidth=config["disentangle"]["bandwidth"],
        #     )
        # else:
        #     train_func = functools.partial(train_epoch)

        starttime = time.time()
        train_metrics, z_train = train_epoch(
            config=config,
            model=model,
            loader=train_loader,
            optimizer=optimizer,
            scheduler=scheduler,
            device="cuda",
            epoch=epoch,
        )
        metrics = {"{}_train".format(k): v for k, v in train_metrics.items()}

        if "grad_reversal" in model.disentangle.keys():
            for key in model.disentangle["grad_reversal"].keys():
                model.disentangle["grad_reversal"][key].reset_parameters()

        if "moving_avg_lsq" in model.disentangle.keys():
            for key in model.disentangle["moving_avg_lsq"].keys():
                metrics["lambda_mals_{}".format(key)] = (
                    model.disentangle["moving_avg_lsq"][key].lam1.detach().cpu().numpy()
                )

        if "qda" in model.disentangle.keys():
            for key in model.disentangle["qda"].keys():
                metrics["lambda_qda_{}".format(key)] = (
                    model.disentangle["qda"][key].lama.detach().cpu().numpy()
                )

        metrics["time"] = time.time() - starttime
        # epoch_loss["epoch"] = epoch
        # loss_dict = {k: v + [epoch_loss[k]] for k, v in loss_dict.items()}

        if epoch % 5 == 0:
            # rand_state = torch.random.get_rng_state()
            # print(rand_state)
            # import pdb; pdb.set_trace()
            # torch.manual_seed(100)
            test_metrics, z_test = test_epoch(
                config=config,
                model=model,
                loader=test_loader,
                device="cuda",
                epoch=epoch,
            )
            metrics.update({"{}_test".format(k): v for k, v in test_metrics.items()})

            for key in ["avg_speed_3d", "heading"]:
                y_true = test_loader.dataset[:][key].detach().cpu().numpy()
                r2_lin = linear_rand_cv(
                    z_test,
                    y_true,
                    int(np.ceil(model.window / config["data"]["stride"])),
                    5,
                )
                r2_mlp = mlp_rand_cv(
                    z_test,
                    y_true,
                    int(np.ceil(model.window / config["data"]["stride"])),
                    5,
                )
                metrics["r2_{}_lin_mean".format(key)] = np.mean(r2_lin)
                metrics["r2_{}_lin_std".format(key)] = np.std(r2_lin)
                metrics["r2_{}_mlp_mean".format(key)] = np.mean(r2_mlp)
                metrics["r2_{}_mlp_std".format(key)] = np.std(r2_mlp)

            z_scaled = StandardScaler().fit_transform(z_train)
            y_true = (
                train_loader.dataset[:]["ids"].detach().cpu().numpy().astype(np.int)
            )
            acc_log = log_class_rand_cv(
                z_scaled,
                y_true,
                int(np.ceil(model.window / config["data"]["stride"])),
                5,
            )
            acc_qda = qda_rand_cv(
                z_scaled,
                y_true,
                int(np.ceil(model.window / config["data"]["stride"])),
                5,
            )
            metrics["acc_ids_log_mean"] = np.mean(acc_log)
            metrics["acc_ids_log_std"] = np.std(acc_log)
            metrics["acc_ids_qda_mean"] = np.mean(acc_qda)
            metrics["acc_ids_qda_std"] = np.std(acc_qda)

            import pdb

            pdb.set_trace()

            vanilla_gmm = np.load(
                "/mnt/home/jwu10/working/ceph/results/vae/vanilla_64/2/vis_latents_train/z_300_gmm.npy"
            )

            walking_list = [1, 4, 8, 38, 41, 44]

            cluster.gmm(
                latents=z_test,
                n_components=50,
                label="".format(epoch),
                covariance_type="diag" if config["model"]["diag"] else "full",
                path = "{}/clusters/".format(config["out_path"])
            )

            # torch.random.set_rng_state(rand_state)

            # metrics.update({"{}_test".format(k):v for k,v in test_loss.items()})
            # run = wandb.Api().run("joshuahwu/wandb_test/{}".format(wandb_run.))
            # wandb_run.run.history().to_csv("metrics.csv")

            print("Saving model to folder: {}".format(config["out_path"]))
            torch.save(
                {k: v.cpu() for k, v in model.state_dict().items()},
                "{}/weights/epoch_{}.pth".format(config["out_path"], epoch),
            )

            if epoch % 20 == 0:
                torch.save(
                    {"optimizer": optimizer.state_dict(), "lr_scheduler": scheduler},
                    "{}/checkpoints/epoch_{}.pth".format(config["out_path"], epoch),
                )

            # pickle.dump(
            #     loss_dict,
            #     open("{}/losses/loss_dict_Train.p".format(config["out_path"]), "wb"),
            # )

            # plt_loss(loss_dict, config["out_path"], config["disentangle"]["features"])

        wandb.log(metrics, epoch)

    return model
