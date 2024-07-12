import numpy as np
import matplotlib.pyplot as plt
import ssumo
import pickle
from scipy.spatial.distance import cdist, pdist
from scipy import stats

# from base_path import RESULTS_PATH, CODE_PATH
import sys
from pathlib import Path
from neuroposelib import read
from sklearn.decomposition import PCA
import colorcet as cc
from sklearn.model_selection import KFold
from sklearn.preprocessing import StandardScaler
import tqdm
from sklearn.linear_model import LogisticRegression
from sklearn.discriminant_analysis import QuadraticDiscriminantAnalysis
from sklearn.metrics import roc_auc_score
from ssumo.eval.metrics import (
    log_class_rand_cv,
    qda_rand_cv,
    linear_rand_cv,
    mlp_rand_cv,
)
import torch


# plt.rcParams["text.usetex"] = True
# plt.rcParams["font.family"] = "serif"
plt.rcParams["font.size"] = "10"
plt.rcParams["font.family"] = "sans-serif"
plt.rcParams["font.sans-serif"] = "Arial"
CODE_PATH = "/mnt/home/jwu10/working/ssumo/"
RESULTS_PATH = "/mnt/ceph/users/jwu10/results/vae/"


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

titles = {
    "pd_speed": "Average Speed",
    "pd_ids": "Animal ID",
}

f = plt.figure(figsize=(15, 15))
gs = f.add_gridspec(4, 8)

for var_ind, var_key in enumerate(["pd_speed", "pd_ids"]):
    m_config = read.config(CODE_PATH + "configs/exp_finals.yaml")[var_key]
    m_dict = {m[0]: [m[1], m[2]] for m in m_config}

    if var_ind == 0:
        config = read.config(
            RESULTS_PATH + m_dict["C-VAE"][0] + "/model_config.yaml"
        )
        loader = ssumo.get.mouse_data(
            data_config=config["data"],
            window=config["model"]["window"],
            train=True,
            data_keys=[
                "x6d",
                "root",
                "avg_speed_3d",
                "fluorescence",
                "ids",
            ],
            shuffle=False,
            normalize=["avg_speed_3d"],
        )
        pd_label = np.array(loader.dataset[:]["fluorescence"] < 0.9, dtype=int).ravel()
        speed = loader.dataset[:]["avg_speed_3d"].cpu().detach().numpy()
        fluorescence = loader.dataset[:]["fluorescence"].cpu().detach().numpy().ravel()
        ids = loader.dataset[:]["ids"].cpu().detach().numpy().ravel()
        not_id = 30
        discrete_classes = np.unique(ids)[np.unique(ids) != not_id]
        pd_label = pd_label[ids != not_id]
        fluorescence = fluorescence[ids != not_id]
        speed = speed[ids != not_id]

    qda, lc, lin, mlp, mmd, mmd_id = {}, {}, {}, {}, {}, {}
    print(var_key)
    for m_key in m_dict:
        path = "{}{}/".format(RESULTS_PATH, m_dict[m_key][0])
        z = np.load(path + "latents/Train_{}.npy".format(m_dict[m_key][1]))
        z = z[ids != not_id]
        # z = StandardScaler().fit_transform(z)

        if var_key == "pd_speed":
            lc[m_key] = log_class_rand_cv(z, pd_label, 51, 5)
            qda[m_key] = qda_rand_cv(z, pd_label, 51, 5)
            lin[m_key] = linear_rand_cv(z, speed, 51, 5)
            mlp[m_key] = mlp_rand_cv(z, speed, 51, 5)

        if var_key == "pd_ids":
            window = 51
            qda[m_key], lc[m_key] = [], []
            kf = KFold(n_splits=len(discrete_classes), shuffle=False)
            # for i, (train_i, test_i) in enumerate(kf.split(discrete_classes)):
            test_i = [17,22,21]
            for i, train_i in enumerate(discrete_classes[~np.isin(discrete_classes,test_i)]):
                print(train_i)
                z_train = z[np.isin(ids[ids != not_id], [test_i])][# [discrete_classes])][#[test_i])][
                    ::window
                ]
                # print(discrete_classes[test_i], print(discrete_classes[train_i]))
                y_train = pd_label[
                    np.isin(ids[ids != not_id], [test_i])#[discrete_classes])#[test_i])
                ][::window].ravel()
                z_test = z[np.isin(ids[ids != not_id], train_i)][#])][#discrete_classes[train_i])][
                    ::window
                ]
                # y_test = pd_label[np.isin(ids[ids!=not_id], discrete_classes[train_i])][::window]
                y_test = np.zeros((len(z_test), 2))

                y_test[
                    np.arange(len(z_test)),
                    pd_label[np.isin(ids[ids != not_id], train_i)][#discrete_classes[train_i])][
                        ::window
                    ],
                ] = 1

                clf = QuadraticDiscriminantAnalysis().fit(z_train, y_train)
                # y_pred = clf.predict(z_test)
                # acc = (y_test == y_pred).sum()/len(y_test)
                # qda[m_key] += [acc]
                assert (clf.classes_ == np.array([0,1])).sum() == 2
                qda[m_key] += [roc_auc_score(y_test, clf.predict_proba(z_test))]

                clf = LogisticRegression(solver="sag", max_iter=300, C=0.5).fit(
                    z_train, y_train
                )
                lc[m_key] += [roc_auc_score(y_test, clf.predict_proba(z_test))]
                # y_pred = clf.predict(z_test)
                # acc = (y_test == y_pred).sum()/len(y_test)
                # lc[m_key] += [acc]

            mmd[m_key] = []
            for i in discrete_classes:
                pd_i = ((fluorescence < 0.9) & (ids[ids!=not_id] == i)).ravel()
                healthy_i = ((fluorescence >= 0.9) & (
                    ids[ids!=not_id] == i
                )).ravel()

                pd_speed_i = speed[pd_i, ...]
                healthy_speed_i = speed[healthy_i, ...]
                pd_z_i = z[pd_i, ...]
                healthy_z_i = z[healthy_i, ...]
                mmd[m_key] += [mmd_estimate(pd_z_i[::window], healthy_z_i[::window])]

            mmd_id[m_key] = []
            for i in tqdm.tqdm(range(len(discrete_classes)-1)):
                for j in range(i+1, len(discrete_classes)):
                    z1 = z[ids[ids!=not_id] == discrete_classes[i]][::window]
                    z2 = z[ids[ids!=not_id] == discrete_classes[j]][::window]

                    mmd_id[m_key] += [mmd_estimate(z1, z2)]

    if var_key == "pd_speed":
        lc["Speed Only"] = log_class_rand_cv(speed, pd_label, 51, 5)
        qda["Speed Only"] = qda_rand_cv(speed, pd_label, 51, 5)

    ### Plot 5 Fold R2 Decoding
    bar_ax = f.add_subplot(gs[var_ind, :4])
    bar_ax.set_title("Parkinson's Prediction from Latents with {} Scrubbing".format(titles[var_key]),fontsize=14)
    w = 0.25  # bar width
    x = np.arange(len(lc.keys())) + 0.33  # x-coordinates of your bars
    # colors = [(0, 0, 1, 1), (1, 0, 0, 1)]    # corresponding colors

    bar_ax.bar(
        x,
        height=[np.mean(lc[k]) for k in lc.keys()],
        width=w,  # bar width
        # tick_label=list(lc.keys()),
        color = "#9871bb",
        label="Logistic",
    )
    print(lc)
    print(qda)

    # for i in range(len(discrete_classes)):
    #     bar_ax.plot(x, [v for k, v in lc.items()], lw=0.5)

    ### PD Decoding
    for i, key in enumerate(lc.keys()):
        # distribute scatter randomly across whole width of bar
        bar_ax.scatter(x[i] + np.random.uniform(-0.1, 0.1,len(lc[key])), lc[key], marker="o", c="k", s=1)

    bar_ax.bar(
        x + 0.33,
        height=[np.mean(qda[k]) for k in qda.keys()],
        width=w,  # bar width
        # tick_label=list(qda.keys()),
        color= "#d05873",
        label="QDA",
    )
    for i, key in enumerate(qda.keys()):
        # distribute scatter randomly across whole width of bar
        bar_ax.scatter(
            x[i] + 0.33 + np.random.uniform(-0.1, 0.1,len(qda[key])), qda[key], marker="o", c="k", s=1
        )

    print({k:np.mean(qda[k]) for k in lc.keys()})

    # for i in range(len(discrete_classes)):
    #     bar_ax.plot(x, [v for k, v in qda.items()], lw=0.5)

    bar_ax.set_xticks(x + 0.33 / 2)
    bar_ax.set_xticklabels(list(lc.keys()))
    bar_ax.set_ylabel("AUROC")
    bar_ax.legend()

    if var_key == "pd_speed":
        bar_ax = f.add_subplot(gs[var_ind, 4:])
        x = np.arange(len(lin.keys())) + 0.33
        ### Speed Decoding
        bar_ax.bar(
            x,
            height=[np.mean(lin[k]) for k in lin.keys()],
            width=w,  # bar width
            # tick_label=list(lc.keys()),
            color = "#009392",
            label="Linear",
        )

        for i, key in enumerate(lin.keys()):
            # distribute scatter randomly across whole width of bar
            bar_ax.scatter(x[i] + np.random.uniform(-0.1, 0.1,len(lin[key])), lin[key], marker="o", c="k", s=1)

        bar_ax.bar(
            x + 0.33,
            height=[np.mean(mlp[k]) for k in mlp.keys()],
            width=w,  # bar width
            color = "#028bc3",
            # tick_label=list(qda.keys()),
            label="Nonlinear (MLP)",
        )
        for i, key in enumerate(mlp.keys()):
            # distribute scatter randomly across whole width of bar
            bar_ax.scatter(
                x[i] + 0.33 + np.random.uniform(-0.1, 0.1,len(mlp[key])), mlp[key], marker="o", c="k", s=1
            )

        bar_ax.set_xticks(x + 0.33 / 2)
        bar_ax.set_xticklabels(list(mlp.keys()))
        bar_ax.set_ylabel(r"$R^2$")
        bar_ax.legend()

    if var_key == "pd_ids":
        bar_ax = f.add_subplot(gs[var_ind+1, :4])
        x = np.arange(len(mmd.keys())) + 0.33
        ### Speed Decoding
        bar_ax.bar(
            x,
            height=[np.mean(mmd[k]) for k in mmd.keys()],
            width=w,  # bar width
            # tick_label=list(lc.keys()),
            label="MMD",
        )

        for i, key in enumerate(mmd.keys()):
            # distribute scatter randomly across whole width of bar
            bar_ax.scatter(x[i] + np.random.uniform(-0.1, 0.1,len(mmd[key])), mmd[key], marker="o", c="k", s=1)

        bar_ax.set_xticks(x)
        bar_ax.set_xticklabels(list(mmd.keys()))
        bar_ax.set_ylabel("MMD")
        bar_ax.set_title("Maximum Mean Discrepancy Between Healthy and PD", fontsize=15)
        # bar_ax.legend()

        for i, k in enumerate(mmd.keys()):
            print(mmd)
            # import pdb; pdb.set_trace()
            ax = f.add_subplot(gs[var_ind+2,i*2:i*2+2])
            idx = np.unique(fluorescence, return_index=True)[1]
            ax.scatter(fluorescence[np.sort(idx)][:-1], mmd[k])
            ax.set_xlabel("Integrated Fluorescence")
            ax.set_ylabel("MMD")
            ax.set_title("{}\nPearson: {:3f}".format(k, stats.pearsonr(fluorescence[np.sort(idx)][:-1],mmd[k])[0]))

            # ax2 = f.add_subplot(gs[var_ind+1,4:])
        bar_ax = f.add_subplot(gs[var_ind+1,4:])
        x = np.arange(len(mmd_id.keys()))
        ### Speed Decoding
        bar_ax.bar(
            x,
            height=[np.mean(mmd_id[k]) for k in mmd_id.keys()],
            width=w,  # bar width
            # tick_label=list(lc.keys()),
            label="MMD",
        )

        for i, key in enumerate(mmd_id.keys()):
            print(mmd_id)
            # distribute scatter randomly across whole width of bar
            bar_ax.scatter(
                x[i] + np.random.uniform(-0.1,0.1,len(mmd_id[key])), mmd_id[key], marker="o", c="k", s=0.5, alpha=1/3
            )

        bar_ax.set_xticks(x)
        bar_ax.set_xticklabels(list(mmd_id.keys()))
        bar_ax.set_ylabel("MMD")
        bar_ax.set_title("Maximum Mean Discrepancy Between Individuals", fontsize=15)
        # bar_ax.legend()


f.tight_layout()
plt.savefig("./results/pd_final.png")