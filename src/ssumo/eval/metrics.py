import numpy as np
from pathlib import Path
from neuroposelib import read
from ssumo import get
from . import project_to_null
from sklearn.metrics import r2_score
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.discriminant_analysis import QuadraticDiscriminantAnalysis
from sklearn.preprocessing import StandardScaler
import pickle
import functools
from ..model.disentangle import MLP
import torch.optim as optim
import torch
from tqdm import trange
from sklearn.model_selection import cross_val_score, cross_val_predict, KFold
from math import ceil


def epoch_regression(
    path,
    method,
    dataset_label,
    save_load=True,
    disentangle_keys=["avg_speed", "heading", "heading_change"],
):
    label = method + "_reg"
    config = read.config(path + "/model_config.yaml")
    config["model"]["load_model"] = config["out_path"]
    config["data"]["project_axis"] = 1
    # config["data"]["stride"] = 50

    pickle_path = "{}/{}_{}.p".format(config["out_path"], label, dataset_label)
    if Path(pickle_path).is_file() and save_load:
        metrics = pickle.load(open(pickle_path, "rb"))
        epochs_to_test = [
            e
            for e in get.all_saved_epochs(path)
            if (e not in metrics["epochs"])  # and (e>100)
        ]
        metrics["epochs"] = np.concatenate([metrics["epochs"], epochs_to_test]).astype(
            int
        )
    else:
        if ("log_class" in method) or ("qda" in method):
            metrics = {k: {"Accuracy": []} for k in disentangle_keys}
        elif "_cv" in method:
            metrics = {k: {"R2": []} for k in disentangle_keys}
        else:
            metrics = {k: {"R2": [], "R2_Null": []} for k in disentangle_keys}
        metrics["epochs"] = [e for e in get.all_saved_epochs(path)]  # if (e>100)]
        epochs_to_test = metrics["epochs"]

    data_keys = ["x6d", "root"]
    data_keys += ["offsets"] if config["data"].get("segment_lens") else []
    # data_keys = []
    # data_keys += ["ids"] if "_cv" in method else []

    if len(epochs_to_test) > 0:
        loader = get.mouse_data(
            config=config,
            window=config["model"]["window"],
            train=dataset_label == "Train",
            data_keys=data_keys + disentangle_keys,
            shuffle=False,
            normalize=[
                d for d in disentangle_keys if d not in ["heading", "ids", "view_axis"]
            ],
        )

    for _, epoch in enumerate(epochs_to_test):
        model = get.model(
            config=config,
            load_model=config["out_path"],
            epoch=epoch,
            n_keypts=loader.dataset.n_keypts,
            direction_process=config["data"]["direction_process"],
            arena_size=loader.dataset.arena_size,
            kinematic_tree=loader.dataset.kinematic_tree,
            bound=config["data"]["normalize"] is not None,
            discrete_classes=loader.dataset.discrete_classes,
            verbose=-1,
        )

        z = get.latents(
            config, model, epoch, loader, "cuda", dataset_label, recompute=False
        )

        for key in disentangle_keys:
            print("Decoding Feature: {}".format(key))
            if key == "ids":
                y_true = loader.dataset[:][key].detach().cpu().numpy().astype(np.int)
            if key == "view_axis":
                y_true = loader.dataset[:][key][:, 1:]
                # y_true = np.abs(np.arctan2(y_true[:, 1], y_true[:, 2]))[:, None]
            else:
                y_true = loader.dataset[:][key].detach().cpu().numpy()

            if method == "log_class":
                accuracy = log_class_regression(z, y_true)
                metrics[key]["Accuracy"] += [accuracy]
            elif "_cv" in method:
                if method == "linear_cv":
                    r2 = linear_cv(
                        z,
                        y_true,
                        loader.dataset[:]["ids"].detach().cpu().numpy().ravel(),
                    )
                    metrics[key]["R2"] += [r2]
                elif method == "mlp_cv":
                    r2 = mlp_cv(
                        z,
                        y_true,
                        loader.dataset[:]["ids"].detach().cpu().numpy().ravel(),
                    )
                    metrics[key]["R2"] += [r2]
                elif method == "log_class_cv":
                    acc = log_class_cv(
                        z,
                        y_true,
                        loader.dataset[:]["ids"].detach().cpu().numpy().ravel(),
                    )

                    metrics[key]["Accuracy"] += [acc]

                elif method == "linear_rand_cv":
                    r2 = linear_rand_cv(z, y_true, model.window, 5)
                    metrics[key]["R2"] += [r2]

                elif method == "mlp_rand_cv":
                    r2 = mlp_rand_cv(z, y_true, model.window, 5)
                    metrics[key]["R2"] += [r2]

                elif method == "log_class_rand_cv":
                    acc = log_class_rand_cv(
                        StandardScaler().fit_transform(z), y_true, model.window, 5
                    )
                    metrics[key]["Accuracy"] += [acc]

                elif method == "qda_rand_cv":
                    acc = qda_rand_cv(
                        StandardScaler().fit_transform(z), y_true, model.window, 5
                    )
                    print(metrics[key])
                    metrics[key]["Accuracy"] += [acc]

            else:
                if method == "linear":
                    r2, r2_null = linear_regression(z, y_true, model, key)
                elif method == "mlp":
                    r2, r2_null = mlp_regression(z, y_true, model, key)

                metrics[key]["R2_Null"] += [r2_null]
                metrics[key]["R2"] += [r2]

    print(metrics)

    if save_load:
        pickle.dump(
            metrics,
            open(pickle_path, "wb"),
        )

    return metrics


def log_class_regression(z, y_true):
    LR_Classifier = LogisticRegression(
        multi_class="multinomial", solver="sag", max_iter=200
    ).fit(z, y_true.ravel())
    pred = LR_Classifier.predict(z)
    accuracy = (y_true.ravel() == pred).sum() / len(y_true)
    return accuracy


def linear_regression(z, y_true, model, key):
    lin_model = LinearRegression().fit(z, y_true)
    pred = lin_model.predict(z)

    r2 = r2_score(y_true, pred)
    if "linear" in model.disentangle.keys():
        if key in model.disentangle["linear"].keys():
            dis_w = (
                model.disentangle["linear"][key].decoder.weight.detach().cpu().numpy()
            )
    else:
        dis_w = lin_model.coef_
        # z -= lin_model.intercept_[:,None] * dis_w

    ## Null space projection
    z_null = project_to_null(z, dis_w)[0]
    pred_null = LinearRegression().fit(z_null, y_true).predict(z_null)
    r2_null = r2_score(y_true, pred_null)
    return r2, r2_null


def custom_cv_5folds(i, ids, folds=5):
    full_ind = np.arange(len(ids), dtype=int)
    idx = []
    for id in np.unique(ids):
        id_idx = full_ind[ids == id]
        id_split = np.linspace(0, len(id_idx), folds + 1).astype(int)
        idx += [id_idx[id_split[i] : id_split[i + 1]]]

    idx_test = np.concatenate(idx, axis=0)
    idx_train = full_ind[~np.isin(full_ind, idx_test)]
    return idx_train, idx_test


def linear_rand_cv(z, y_true, window=51, folds=5):
    r2 = []
    for shift_i in range(2):
        start_i = shift_i * (window // 2)
        downsampled_z = z[start_i::window, ...]
        downsampled_y = y_true[start_i::window, ...]
        kf = KFold(n_splits=folds, shuffle=True)
        for i, (train_i, test_i) in enumerate(kf.split(downsampled_z)):
            clf = LinearRegression().fit(downsampled_z[train_i], downsampled_y[train_i])
            y_pred = clf.predict(downsampled_z[test_i])
            r2 += [r2_score(downsampled_y[test_i], y_pred)]

    print("Train Length: {}, Test Length: {}".format(len(train_i), len(test_i)))
    return r2


def log_class_rand_cv(z, y_true, window=51, folds=5):
    accuracy = []
    for shift_i in range(2):
        start_i = shift_i * (window // 2)
        downsampled_z = z[start_i::window, ...]
        downsampled_y = y_true[start_i::window, ...]
        kf = KFold(n_splits=folds, shuffle=True)
        for i, (train_i, test_i) in enumerate(kf.split(downsampled_z)):
            clf = LogisticRegression(
                multi_class="multinomial", solver="sag", max_iter=300
            ).fit(downsampled_z[train_i], downsampled_y[train_i].ravel())
            y_pred = clf.predict(downsampled_z[test_i])
            acc = (downsampled_y[test_i].ravel() == y_pred).sum() / len(test_i)

            accuracy += [acc]

    print("Train Length: {}, Test Length: {}".format(len(train_i), len(test_i)))
    return accuracy


def qda_rand_cv(z, y_true, window=51, folds=5):
    accuracy = []
    for shift_i in range(2):
        start_i = shift_i * (window // 2)
        downsampled_z = z[start_i::window, ...]
        downsampled_y = y_true[start_i::window, ...]
        kf = KFold(n_splits=folds, shuffle=True)
        for i, (train_i, test_i) in enumerate(kf.split(downsampled_z)):
            clf = QuadraticDiscriminantAnalysis().fit(
                downsampled_z[train_i], downsampled_y[train_i].ravel()
            )
            y_pred = clf.predict(downsampled_z[test_i])
            acc = (downsampled_y[test_i].ravel() == y_pred).sum() / len(test_i)
            accuracy += [acc]
    print("Train Length: {}, Test Length: {}".format(len(train_i), len(test_i)))
    return accuracy


def mlp_rand_cv(z, y_true, window=51, folds=5):
    r2 = []
    for shift_i in range(2):
        start_i = shift_i * (window // 2)
        downsampled_z = z[start_i::window, ...]
        downsampled_y = y_true[start_i::window, ...]
        kf = KFold(n_splits=folds, shuffle=True)
        for i, (train_i, test_i) in enumerate(kf.split(downsampled_z)):
            model = train_MLP(
                torch.tensor(downsampled_z[train_i]), downsampled_y[train_i], 200
            )[0]
            y_pred = (
                model(torch.tensor(downsampled_z[test_i]).cuda()).cpu().detach().numpy()
            )
            r2 += [r2_score(downsampled_y[test_i], y_pred)]
    print("Train Length: {}, Test Length: {}".format(len(train_i), len(test_i)))
    return r2


def linear_cv(z, y_true, ids, folds=5):
    r2 = []
    for i in range(folds):
        idx_train, idx_test = custom_cv_5folds(i, ids)
        pred = (
            LinearRegression().fit(z[idx_train], y_true[idx_train]).predict(z[idx_test])
        )
        r2 += [r2_score(y_true[idx_test], pred)]

    return r2


def mlp_cv(z, y_true, ids, folds=5):
    r2 = []
    for i in range(folds):
        idx_train, idx_test = custom_cv_5folds(i, ids)
        model = train_MLP(z[idx_train], y_true[idx_train], 200)[0]
        pred = model(torch.tensor(z[idx_test]).cuda()).cpu().detach().numpy()
        r2 += [r2_score(y_true[idx_test], pred)]

    return r2


def log_class_cv(z, y_true, ids, folds=5):
    acc = []
    for i in range(folds):
        idx_train, idx_test = custom_cv_5folds(i, ids)
        clf = LogisticRegression(
            multi_class="multinomial", solver="sag", max_iter=300
        ).fit(z[idx_train], y_true[idx_train].ravel())

        accuracy = (y_true[idx_test].ravel() == clf.predict(z[idx_test])).sum() / len(
            idx_test
        )

        acc += [accuracy]

    return acc


def mlp_regression(z, y_true, model, key):
    pred = train_MLP(z, y_true, 200)[1]
    r2 = r2_score(y_true, pred)

    if "linear" in model.disentangle.keys():
        if key in model.disentangle["linear"].keys():
            dis_w = (
                model.disentangle["linear"][key].decoder.weight.detach().cpu().numpy()
            )
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

    print("Loss: {}".format(loss.item() / len(y_true)))

    model.eval()
    y_pred = model(z)

    return model, y_pred.detach().cpu().numpy()
