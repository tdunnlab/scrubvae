import numpy as np
from pathlib import Path
from neuroposelib import read
from scrubvae import get
from . import project_to_null
from sklearn.metrics import r2_score
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.discriminant_analysis import QuadraticDiscriminantAnalysis, LinearDiscriminantAnalysis
from sklearn.preprocessing import StandardScaler
import pickle
from ..model.disentangle import MLP
import torch.optim as optim
import torch
from sklearn.model_selection import KFold
from scipy.spatial.distance import cdist, pdist
import functools
from ..eval import cluster
from sklearn.mixture import GaussianMixture
from pandas import crosstab
from scipy.optimize import linear_sum_assignment
import numpy.typing as npt

def epoch_metric(func):
    @functools.wraps(func)
    def wrapper(
        path,
        method,
        dataset_label,
        save_load=True,
        disentangle_keys=["avg_speed", "heading"],
        start_epoch=100,
        **kwargs,
    ):
        config = read.config(path + "/model_config.yaml")
        config["model"]["load_model"] = config["out_path"]

        pickle_path = "{}/{}_{}.p".format(config["out_path"], method, dataset_label)
        if Path(pickle_path).is_file() and save_load:
            metrics = pickle.load(open(pickle_path, "rb"))
            epochs_to_test = [
                e
                for e in get.all_saved_epochs(path)
                if (e not in metrics["epochs"]) and (e > start_epoch)
            ]
            metrics["epochs"] = np.concatenate(
                [metrics["epochs"], epochs_to_test]
            ).astype(int)
        else:
            metrics = {
                "epochs": [e for e in get.all_saved_epochs(path) if (e > start_epoch)]
            }
            epochs_to_test = metrics["epochs"]

        data_keys = ["x6d", "root"]
        data_keys += (
            ["ids"] if method in ["linear_cv", "mlp_cv", "log_class_cv"] else []
        )

        if len(epochs_to_test) > 0:
            loader = get.mouse_data(
                data_config=config["data"],
                train_val_test = dataset_label,
                data_keys=data_keys + disentangle_keys,
                shuffle=False,
            )

            metrics = func(
                config=config,
                loader=loader,
                epochs_to_test=epochs_to_test,
                metrics=metrics,
                dataset_label=dataset_label,
                disentangle_keys=disentangle_keys,
                method=method,
                **kwargs,
            )

        print(metrics)

        if save_load:
            pickle.dump(
                metrics,
                open(pickle_path, "wb"),
            )

        return metrics

    return wrapper


@epoch_metric
def epoch_cluster_entropy(
    config,
    loader,
    epochs_to_test,
    metrics,
    dataset_label,
    comparison_clustering,
    n_components,
    **kwargs,
):
    if "Entropy" not in metrics.keys():
        metrics["Entropy"] = []
    k_preds0 = np.load(comparison_clustering)
    assert len(loader.dataset) == len(k_preds0)

    for _, epoch in enumerate(epochs_to_test):
        model = get.model(
            model_config=config["model"],
            load_model=config["out_path"],
            epoch=epoch,
            disentangle_config=config["disentangle"],
            loss_config=config["loss"],
            n_keypts=loader.dataset.n_keypts,
            direction_process=config["data"]["direction_process"],
            arena_size=loader.dataset.arena_size,
            kinematic_tree=loader.dataset.kinematic_tree,
            bound=config["data"]["normalize"] is not None,
            discrete_classes=loader.dataset.discrete_classes,
            verbose=-1,
        )
        z = get.latents(config, model, epoch, loader, "cuda", dataset_label)

        k_preds1 = GaussianMixture(
            n_components=n_components,
            covariance_type="diag" if config["model"]["diag"] else "full",
            max_iter=150,
            init_params="k-means++",
            reg_covar=1e-5,
            verbose=1,
        ).fit_predict(z)

        entropy = 0
        for i in range(n_components):
            hist = (
                np.histogram(
                    k_preds0[k_preds1 == i],
                    bins=np.arange(k_preds0.max() + 2) - 0.5,
                )[0]
                / (k_preds1 == i).sum()
            )

            entropy += np.nan_to_num(hist * np.log2(1 / hist)).sum()

        metrics["Entropy"] += [entropy / n_components]

    return metrics


@epoch_metric
def epoch_regression(
    config: dict,
    loader,
    epochs_to_test,
    metrics,
    method,
    dataset_label,
    disentangle_keys=["avg_speed", "heading", "heading_change"],
):
    stride = 1 if config["data"]["dataset"] == "4_mice" else 10
    print("Stride {stride}")
    if len(metrics.keys()) == 1:
        if ("log_class" in method) or ("qda" in method):
            metrics.update({k: {"Accuracy": []} for k in disentangle_keys})
        elif "_cv" in method:
            metrics.update({k: {"R2": []} for k in disentangle_keys})
        else:
            metrics.update({k: {"R2": [], "R2_Null": []} for k in disentangle_keys})

    for _, epoch in enumerate(epochs_to_test):
        model = get.model(
            model_config=config["model"],
            load_model=config["out_path"],
            epoch=epoch,
            disentangle_config=config["disentangle"],
            loss_config=config["loss"],
            n_keypts=loader.dataset.n_keypts,
            direction_process=config["data"]["direction_process"],
            arena_size=loader.dataset.arena_size,
            kinematic_tree=loader.dataset.kinematic_tree,
            bound=config["data"]["normalize"] is not None,
            discrete_classes=loader.dataset.discrete_classes,
            verbose=-1,
        )

        z = get.latents(config, model, epoch, loader, "cuda", dataset_label)

        for key in disentangle_keys:
            print("Decoding Feature: {}".format(key))
            if key == "ids":
                y_true = loader.dataset[:][key].detach().cpu().numpy().astype(int)
            else:
                y_true = loader.dataset[:][key].detach().cpu().numpy()

            if method == "linear_rand_cv":
                r2 = linear_rand_cv(z, y_true, model.window, 5)
                metrics[key]["R2"] += [r2]

            elif method == "mlp_rand_cv":
                r2 = mlp_rand_cv(z, y_true, model.window, 5)
                metrics[key]["R2"] += [r2]

            elif method == "log_class_rand_cv":
                acc = log_class_rand_cv(
                    z, y_true, model.window//stride, 5
                )
                metrics[key]["Accuracy"] += [acc]

            elif method == "qda_rand_cv":
                acc = qda_rand_cv(
                    z, y_true, model.window//stride, 5
                )
                print(metrics[key])
                metrics[key]["Accuracy"] += [acc]

    return metrics

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


def rand_cv(func):
    @functools.wraps(func)
    def wrapper(
        z,
        y_true,
        window=51,
        folds=5,
        **kwargs,
    ):
        met = []
        for shift_i in range(1):
            start_i = shift_i * (window // 2)
            downsampled_z = z[start_i::window, ...]
            downsampled_y = y_true[start_i::window, ...]
            kf = KFold(n_splits=folds, shuffle=True, random_state=100)
            for i, (train_i, test_i) in enumerate(kf.split(downsampled_z)):
                met += [
                    func(
                        downsampled_z[train_i],
                        downsampled_y[train_i],
                        downsampled_z[test_i],
                        downsampled_y[test_i],
                    )
                ]

        print("Train Length: {}, Test Length: {}".format(len(train_i), len(test_i)))

        return met

    return wrapper


@rand_cv
def linear_rand_cv(z_train, y_train, z_test, y_test):
    clf = LinearRegression().fit(z_train, y_train)
    y_pred = clf.predict(z_test)
    r2 = r2_score(y_test, y_pred)
    return r2


@rand_cv
def log_class_rand_cv(z_train, y_train, z_test, y_test):
    clf = LogisticRegression(
        l1_ratio=0.5,
        penalty="elasticnet",
        multi_class="ovr",
        solver="saga",
        max_iter=300,
    ).fit(z_train, y_train.ravel())
    y_pred = clf.predict(z_test)
    acc = (y_test.ravel() == y_pred).sum() / len(z_test)

    return acc


@rand_cv
def qda_rand_cv(z_train, y_train, z_test, y_test):
    clf = QuadraticDiscriminantAnalysis().fit(z_train, y_train.ravel())
    y_pred = clf.predict(z_test)
    acc = (y_test.ravel() == y_pred).sum() / len(z_test)
    return acc

@rand_cv
def lda_rand_cv(z_train, y_train, z_test, y_test):
    clf = LinearDiscriminantAnalysis().fit(z_train, y_train.ravel())
    y_pred = clf.predict(z_test)
    acc = (y_test.ravel() == y_pred).sum() / len(z_test)
    return acc

@rand_cv
def mlp_rand_cv(z_train, y_train, z_test, y_test):
    model = train_MLP(z_train, y_train, 200)[0]
    y_pred = model(z_test.cuda()).cpu().detach().numpy()
    r2 = r2_score(y_test, y_pred)
    return r2

def train_MLP(z, y_true, num_epochs=200):
    model = MLP(z.shape[-1], y_true.shape[-1]).cuda()
    torch.backends.cudnn.benchmark = True
    z = z.cuda()
    y_true = torch.tensor(y_true, device="cuda")
    optimizer = optim.AdamW(model.parameters(), lr=0.001)
    model.train()
    with torch.enable_grad():
        for epoch in range(num_epochs):
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


def mmd_estimate(X, Y, h=None):
    """
    Given samples from two distributions in a common
    common feature space, this function computes an
    estimate of the maximal mean discrepancy (MMD)
    distance with a squared exponential kernel.

    Reference
    ---------
    Gretton et al. (2012). A Kernel Two-Sample Test.
    Journal of Machine Learning Research 13: 723-773.

    Parameters
    ----------
    X : ndarray (num_x_samples x num_features)
        First set of observed samples, assumed to be
        drawn from some unknown distribution P.

    Y : ndarray (num_y_samples x num_features)
        Second set of observed samples, assumed to be
        drawn from some unknown distribution Q.

    h : float
        Bandwidth parameter

    Returns
    -------
    dist : float
        An unbiased estimator of the MMD.
    """

    # Compute pairwise distances
    xd = pdist(X, metric="euclidean")
    yd = pdist(Y, metric="euclidean")
    xyd = cdist(X, Y, metric="euclidean").ravel()

    if h is None:
        h = np.median(np.concatenate((xd, yd, xyd))) ** 2
    # Compute unbiased MMD distance estimate.
    kxx = np.mean(np.exp(-(xd**2) / h))
    kyy = np.mean(np.exp(-(yd**2) / h))
    kxy = np.mean(np.exp(-(xyd**2) / h))
    return kxx + kyy - 2 * kxy


def shannon_entropy(x):
    counts = np.unique(x, return_counts=True)[1]
    hist = counts / counts.sum()
    entropy = (hist * np.log(1 / hist)).sum()
    return entropy

def shannon_entropy_torch(x, bins, range):
    hist = torch.histogram(x, bins=bins, range=range)[0]
    entropy = torch.nan_to_num(x * torch.log(1 / hist)).sum()
    return entropy

def hungarian_match(x1: npt.ArrayLike, x2: npt.ArrayLike):
    """Matches the categorical values between two sequences using the Hungarian matching algorithm.

    Parameters
    ----------
    x1 : npt.ArrayLike
        Sequence of categorical values.
    x2 : npt.ArrayLike
        Sequence of categorical values.

    Returns
    -------
    mapped_x
        Returns x1 sequence using the matched categorical labels of x2.
    """

    cost = np.array(crosstab(x1, x2))
    row_ind, col_ind = linear_sum_assignment(cost, maximize=True)
    row_k = np.unique(x1)[row_ind]
    col_v = np.unique(x2)[col_ind]
    idx = np.searchsorted(row_k, x1)
    idx[idx == len(row_k)] = 0
    mask = row_k[idx] == x1
    mapped_x = np.where(mask, col_v[idx], x1)
    return mapped_x