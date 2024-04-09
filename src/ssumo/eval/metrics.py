import numpy as np
from pathlib import Path
from dappy import read
from ssumo import get
from . import project_to_null
from sklearn.metrics import r2_score
from sklearn.linear_model import LinearRegression
import pickle
import functools
from ..model.disentangle import MLP, LinearDisentangle
import torch.optim as optim
import torch
from tqdm import trange

def for_all_epochs(func):
    @functools.wraps(func)
    def wrapper(
        path,
        dataset_label,
        save_load=True,
        disentangle_keys=["avg_speed", "heading", "heading_change"],
        **kwargs,
    ):
        if func.__name__ == "epoch_linear_regression":
            label = "lin_reg"
        elif func.__name__ == "epoch_adversarial_attack":
            label = "adv_atk"

        config = read.config(path + "/model_config.yaml")
        config["model"]["load_model"] = config["out_path"]

        pickle_path = "{}/{}_{}.p".format(config["out_path"], label, dataset_label)
        if Path(pickle_path).is_file() and save_load:
            metrics = pickle.load(open(pickle_path, "rb"))
            epochs_to_test = [
                e for e in get.all_saved_epochs(path) if e not in metrics["epochs"]
            ]
            metrics["epochs"] = np.concatenate(
                [metrics["epochs"], epochs_to_test]
            ).astype(int)
        else:
            metrics = {k: {"R2": [], "R2_Null": []} for k in disentangle_keys}
            metrics["epochs"] = get.all_saved_epochs(path)
            epochs_to_test = metrics["epochs"]

        if len(epochs_to_test) > 0:
            dataset = get.mouse_data(
                data_config=config["data"],
                window=config["model"]["window"],
                train=dataset_label == "Train",
                data_keys=[
                    "x6d",
                    "root",
                ]
                + disentangle_keys,
                shuffle=False,
                normalize=disentangle_keys,
            )[0]

        for epoch_ind, epoch in enumerate(epochs_to_test):

            model = get.model(
                model_config=config["model"],
                load_model=config["out_path"],
                epoch=epoch,
                disentangle_config=config["disentangle"],
                n_keypts=dataset.n_keypts,
                direction_process=config["data"]["direction_process"],
                arena_size=dataset.arena_size,
                kinematic_tree=dataset.kinematic_tree,
                bound=config["data"]["normalize"] is not None,
                verbose=-1,
            )

            z = get.latents(config, model, dataset, "cuda", dataset_label)

            for key in disentangle_keys:
                print("Decoding Feature: {}".format(key))

                r2, r2_null = func(z, dataset[:][key].detach().cpu().numpy(), model, key)

                metrics[key]["R2"] += [r2]
                metrics[key]["R2_Null"] += [r2_null]

        print(metrics)

        if save_load:
            pickle.dump(
                metrics,
                open(pickle_path, "wb"),
            )

        return metrics

    return wrapper


@for_all_epochs
def epoch_linear_regression(z, y_true, model, key):
    lin_model = LinearRegression().fit(z, y_true)
    pred = lin_model.predict(z)

    r2 = r2_score(y_true, pred)
    if (key in model.disentangle.keys()) and (isinstance(model.disentangle[key],LinearDisentangle)):
        dis_w = model.disentangle[key].decoder.weight.detach().cpu().numpy()
    else:
        dis_w = lin_model.coef_
        # z -= lin_model.intercept_[:,None] * dis_w

    ## Null space projection
    z_null = project_to_null(z, dis_w)[0]
    pred_null = LinearRegression().fit(z_null, y_true).predict(z_null)

    r2_null = r2_score(y_true, pred_null)

    return r2, r2_null


@for_all_epochs
def epoch_adversarial_attack(z, y_true, model, key):
    pred = train_MLP(z, y_true, 200)[1]
    r2 = r2_score(y_true, pred)
    
    if (key in model.disentangle.keys()) and (isinstance(model.disentangle[key],LinearDisentangle)):
        dis_w = model.disentangle[key].decoder.weight.detach().cpu().numpy()
    else:
        print("No linear decoder - fitting SKLearn Linear Regression")
        lin_model = LinearRegression().fit(z, y_true)
        dis_w = lin_model.coef_

    ## Null space projection
    z_null = project_to_null(z, dis_w)[0]
    pred_null = train_MLP(z_null, y_true, 200)[1]
    r2_null = r2_score(y_true, pred_null)
    return r2, r2_null


def train_MLP(z, y_true, num_epochs=200):
    model = MLP(z.shape[-1], y_true.shape[-1]).cuda()
    torch.backends.cudnn.benchmark = True
    z = z.cuda()
    y_true = torch.tensor(y_true, device="cuda")
    optimizer = optim.Adam(model.parameters(), lr=0.01)
    model.train()
    with torch.enable_grad():
        for epoch in trange(num_epochs):
            for param in model.parameters():
                param.grad = None
            output = model(z)
            loss = torch.nn.MSELoss(reduction="sum")(output, y_true)
            
            loss.backward()
            optimizer.step()

    print("Loss: {}".format(loss.item()/len(y_true)))

    model.eval()
    y_pred = model(z)

    return model, y_pred.detach().cpu().numpy()
