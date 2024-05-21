import numpy as np
import matplotlib.pyplot as plt
import ssumo
import pickle
from scipy.spatial.distance import cdist, pdist
import matplotlib.font_manager
import matplotlib.transforms as mtrans
# from base_path import RESULTS_PATH, CODE_PATH
import sys
from pathlib import Path
from dappy import read
from sklearn.decomposition import PCA
import colorcet as cc
from scipy.stats import circvar
import tqdm

# from matplotlib import rc

# rc('font', **{"family":"sans-serif", "sans-serif":["Arial"]})
# rc("text", usetex=True)

plt.rcParams["font.family"] = "sans-serif"
plt.rcParams["font.sans-serif"] = "Arial"
plt.rcParams["font.size"] = "10"

palette_1 = ["#009392", "#028bc3", "#9871bb", "#d05873"]


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


CODE_PATH = "/mnt/home/jwu10/working/ssumo/"
RESULTS_PATH = "/mnt/ceph/users/jwu10/results/vae/"

titles = {
    "heading": "Heading Direction",
    "avg_speed_3d": "Average Speed",
    "ids": "Animal ID",
}

f = plt.figure(figsize=(16, 8))
gs = f.add_gridspec(3, 7)
for var_ind, var_key in enumerate(["avg_speed_3d", "heading", "ids"]):
    print(var_key)
    models = read.config(CODE_PATH + "configs/exp_finals.yaml")[var_key]
    models = {m[0]: [m[1], m[2]] for m in models}

    if var_ind == 0:
        config = read.config(
            RESULTS_PATH + models["CVAE"][0] + "/model_config.yaml"
        )
        loader = ssumo.get.mouse_data(
            data_config=config["data"],
            window=config["model"]["window"],
            train=True,
            data_keys=["avg_speed_3d", "heading", "ids"],
            shuffle=False,
        )
    var = loader.dataset[:][var_key].numpy()

    downsample = 5
    if var_key == "heading":
        var_plt = np.arctan2(var[:, 1], var[:, 0])
        np.save(config["data"]["data_path"] + "heading_rad.npy", var_plt)
        import pdb; pdb.set_trace()
        cmap = cc.cm["colorwheel"]
    elif var_key == "avg_speed_3d":
        var_plt = var.mean(axis=-1)
        cmap = "viridis"
    elif var_key == "ids":
        var_plt = var.ravel().astype(int)
        permute_ind = np.random.permutation(len(var_plt[::downsample]))
        cmap = ssumo.plot.constants.PALETTE_2

    mlp_cv, lin_cv, lc_cv, qda_cv, gmm_var = {}, {}, {}, {}, {}
    i = 0
    for model in models.keys():
        print(model)
        path = "{}{}/".format(RESULTS_PATH, models[model][0])
        z = np.load(path + "latents/Train_{}.npy".format(models[model][1]))

        if var_key in ["heading", "avg_speed_3d"]:
            try:
                model_lin_cv = pickle.load(
                    open(path + "linear_rand_cv_reg_Train.p", "rb")
                )
                lin_cv[model] = np.array(model_lin_cv[var_key]["R2"])[
                    np.array(model_lin_cv["epochs"]) == models[model][1]
                ].ravel()
            except:
                lin_cv[model] = np.zeros(5)

            try:
                model_mlp_cv = pickle.load(open(path + "mlp_rand_cv_reg_Train.p", "rb"))
                mlp_cv[model] = np.array(model_mlp_cv[var_key]["R2"])[
                    np.array(model_mlp_cv["epochs"]) == models[model][1]
                ].ravel()
            except:
                mlp_cv[model] = np.zeros(5)
        else:
            try:
                model_lc_cv = pickle.load(
                    open(path + "log_class_rand_cv_reg_Train.p", "rb")
                )
                lc_cv[model] = np.array(model_lc_cv[var_key]["Accuracy"])[
                    np.array(model_lc_cv["epochs"]) == models[model][1]
                ].ravel()
            except:
                lc_cv[model] = np.zeros(5)

            try:
                model_qda_cv = pickle.load(open(path + "qda_rand_cv_reg_Train.p", "rb"))
                qda_cv[model] = np.array(model_qda_cv[var_key]["Accuracy"])[
                    np.array(model_qda_cv["epochs"]) == models[model][1]
                ].ravel()
            except:
                qda_cv[model] = np.zeros(5)

        # if var_key == "ids":
        #     n_components = 100
        #     k_pred, gmm = ssumo.eval.cluster.gmm(
        #         latents=z,
        #         n_components=n_components,
        #         label="z{}_{}".format(n_components,models[model][1]),
        #         path=path,
        #         covariance_type="diag",
        #     )
        # else:
        n_components = 25
        k_pred, gmm = ssumo.eval.cluster.gmm(
            latents=z,
            n_components=n_components,
            label="z_{}".format(models[model][1]),
            path=path,
            covariance_type="diag",
        )

        gmm_var[model] = []

        if var_key == "ids":
            for i in tqdm.tqdm(range(3)):
                for j in range(i + 1, 3):
                    gmm_var[model] += [
                        [
                            mmd_estimate(
                                z[var_plt == i, :][::100], z[var_plt == j, :][::100]
                            ),
                            1,
                        ]
                    ]

        else:
            for i in range(n_components):
                if var_key == "heading":
                    gmm_var[model] += [
                        [
                            circvar(var_plt[k_pred == i], high=np.pi, low=-np.pi),
                            (k_pred == i).sum() / len(k_pred) * n_components,
                        ]
                    ]
                elif var_key == "avg_speed_3d":
                    gmm_var[model] += [
                        [
                            np.var(var_plt[k_pred == i]),
                            (k_pred == i).sum() / len(k_pred) * n_components,
                        ]
                    ]
                elif var_key == "ids":
                    hist = (
                        np.histogram(
                            var_plt[k_pred == i],
                            bins=np.arange(var_plt.max() + 2) - 0.5,
                        )[0]
                        / (k_pred == i).sum()
                    )
                    entropy = np.nan_to_num(hist * np.log2(1 / hist))
                    print(entropy)
                    gmm_var[model] += [
                        [
                            entropy.sum(),
                            (k_pred == i).sum() / len(k_pred) * n_components,
                        ]
                    ]

    ### Plot 5 Fold R2 Decoding
    bar_ax = f.add_subplot(gs[var_ind, :4])
    bar_ax.set_title(
        "5-Fold {} Decoding from Latents".format(titles[var_key]),
        family="Arial",
        fontsize=14,
    )
    w = 0.25  # bar width
    x = np.arange(len(models.keys())) + 0.33  # x-coordinates of your bars
    # colors = [(0, 0, 1, 1), (1, 0, 0, 1)]    # corresponding colors

    if var_key in ["heading", "avg_speed_3d"]:
        bar_ax.bar(
            x,
            height=[np.mean(lin_cv[k]) for k in models.keys()],
            width=w,  # bar width
            color = palette_1[0],
            # tick_label=list(lin_cv.keys()),
            label="Linear",
        )
        for i, key in enumerate(lin_cv.keys()):
            # distribute scatter randomly across whole width of bar
            bar_ax.scatter(
                x[i] + np.random.uniform(-0.075,0.075,len(lin_cv[key])), lin_cv[key], marker="o", c="k", s=1
            )

        bar_ax.bar(
            x + 0.33,
            height=[np.mean(mlp_cv[k]) for k in models.keys()],
            width=w,  # bar width
            color = palette_1[1],
            # tick_label=list(mlp_cv.keys()),
            label="MLP",
        )
        for i, key in enumerate(mlp_cv.keys()):
            # distribute scatter randomly across whole width of bar
            bar_ax.scatter(
                x[i] + 0.33 + np.random.uniform(-0.075,0.075,len(mlp_cv[key])), mlp_cv[key], marker="o", c="k", s=1
            )

        # bar_ax.tick_params(axis='x', rotation=14,)
        # trans = mtrans.Affine2D().translate(-20, 0)
        # for t in bar_ax.get_xticklabels():
        #     t.set_transform(t.get_transform()+trans)
        bar_ax.set_xticks(x + 0.33 / 2)
        bar_ax.set_xticklabels(list(lin_cv.keys()))
        bar_ax.set_ylabel(r"$R^2$")
        bar_ax.legend()

    else:
        x = np.arange(len(models.keys())) + 0.33 
        bar_ax.bar(
            x-0.13,
            height=[np.mean(lc_cv[k]) for k in models.keys()],
            width=w,  # bar width
            color = palette_1[2],
            # tick_label=list(lc_cv.keys()),
            label="Logistic",
        )
        for i, key in enumerate(lc_cv.keys()):
            # distribute scatter randomly across whole width of bar
            bar_ax.scatter(
                x[i]-0.13 + np.random.uniform(-0.075,0.075,len(lc_cv[key])), lc_cv[key], marker="o", c="k", s=1
            )
        bar_ax.set_ylabel(r"Accuracy")
        bar_ax.legend()

        bar_ax.bar(
            x + 0.2,
            height=[np.mean(qda_cv[k]) for k in models.keys()],
            width=w,  # bar width
            color = palette_1[3],
            tick_label=list(qda_cv.keys()),
            label="QDA",
        )
        for i, key in enumerate(qda_cv.keys()):
            # distribute scatter randomly across whole width of bar
            bar_ax.scatter(
                x[i] + 0.2 + np.random.uniform(-0.075,0.075,(len(qda_cv[key]))), qda_cv[key], marker="o", c="k", s=1
            )

        bar_ax.hlines(
            0.33,
            xmin=x.min()-0.13 - 0.15,
            xmax=x.max()-0.13 + 0.33+0.15,
            colors="k",
            linestyles="dashed",
            label="Chance",
        )

        
        bar_ax.set_xticks(x-0.13 + 0.33 / 2)
        bar_ax.set_xticklabels(list(lc_cv.keys()))
        # trans = mtrans.Affine2D().translate(-20, 0)
        # for t in bar_ax.get_xticklabels():
        #     t.set_transform(t.get_transform()+trans)
        # bar_ax.tick_params(axis='x', rotation=14,)
        bar_ax.set_ylabel(r"Accuracy")
        bar_ax.legend()

    if var_key == "heading":
        path = "{}{}/".format(RESULTS_PATH, "vanilla_64/1")
        z = np.load(path + "latents/Train_{}.npy".format(300))
        k_pred, gmm = ssumo.eval.cluster.gmm(
            latents=z,
            n_components=n_components,
            label="z_{}".format(300),
            path=path,
            covariance_type="diag",
        )
        gmm_var["Vanilla Processed"] = []
        for i in range(n_components):
            gmm_var["Vanilla Processed"] += [
                        [
                            circvar(var_plt[k_pred == i], high=np.pi, low=-np.pi),
                            (k_pred == i).sum() / len(k_pred) * n_components,
                        ]
                    ]
            
        x = np.arange(len(gmm_var.keys())) + 0.33 

    # Plot Variance
    var_ax = f.add_subplot(gs[var_ind, 4:])
    if var_key == "ids":
        var_ax.set_title("{} Maximum Mean Discrepancy".format(titles[var_key]),fontsize=14)
        var_ax.set_ylabel("MMD")
        means_to_plt = [np.mean(np.prod(gmm_var[k], axis=-1)) for k in gmm_var.keys()]
    else:
        if var_key == "heading":
            var_ax.set_title("{} GMM Cluster Circular Variance".format(titles[var_key]), fontsize=14)
        else:
            var_ax.set_title("{} GMM Cluster Variance".format(titles[var_key]), fontsize=14)
        var_ax.set_ylabel(r"$\sigma^2$")
        means_to_plt = [np.mean(np.prod(gmm_var[k], axis=-1)) for k in gmm_var.keys()]
    # colors = [(0, 0, 1, 1), (1, 0, 0, 1)]    # corresponding colors

    var_ax.bar(
        x,
        height=means_to_plt,
        width=w,
        color = "#DEA1D1" if var_key == "ids" else "#a9a9a9",
        tick_label=list(gmm_var.keys()),
    )
    # var_ax.tick_params(axis='x', rotation=14,)
    # trans = mtrans.Affine2D().translate(-20, 0)
    # for t in var_ax.get_xticklabels():
    #     t.set_transform(t.get_transform()+trans)
    for i, key in enumerate(gmm_var.keys()):
        var_ax.scatter(
            x[i] + np.random.uniform(-0.075,0.075, len(gmm_var[key])),
            np.array(gmm_var[key])[:, 0],
            marker="o",
            c="k", s=1
        )

f.tight_layout()
plt.savefig("./results/metrics_final.png")
