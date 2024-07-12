import numpy as np
import matplotlib.pyplot as plt
import ssumo
import pickle
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
    mmd_estimate
)
import torch

plt.rcParams["font.size"] = "10"
plt.rcParams["font.family"] = "sans-serif"
plt.rcParams["font.sans-serif"] = "Arial"
CODE_PATH = "/mnt/home/jwu10/working/ssumo/"
RESULTS_PATH = "/mnt/ceph/users/jwu10/results/vae/"

titles = {
    "pd_speed": "Average Speed",
    "pd_ids": "Animal ID",
}

f = plt.figure(figsize=(14, 9))
subf = f.subfigures(3,1, height_ratios=[1,1,1.75])

for var_ind, var_key in enumerate(["pd_speed", "pd_ids"]):
    m_config = read.config(CODE_PATH + "configs/exp_finals.yaml")[var_key]
    m_dict = {m[0]: [m[1], m[2]] for m in m_config}

    if var_ind == 0:
        config = read.config(RESULTS_PATH + m_dict["C-VAE"][0] + "/model_config.yaml")
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

    lc, lin, mmd, mmd_id_pd, mmd_id_healthy = {}, {}, {}, {}, {}
    print(var_key)
    for m_key in m_dict:
        path = "{}{}/".format(RESULTS_PATH, m_dict[m_key][0])
        z = np.load(path + "latents/Train_{}.npy".format(m_dict[m_key][1]))
        z = z[ids != not_id]
        # z = StandardScaler().fit_transform(z)

        if var_key == "pd_speed":
            lc[m_key] = log_class_rand_cv(z, pd_label, 51, 5)
            lin[m_key] = linear_rand_cv(z, speed, 51, 5)

            if ("MALS" in m_key) or ("MI" in m_key):
                lc[m_key + "\n+ Speed"] = log_class_rand_cv(
                    np.concatenate([z, speed], axis=-1), pd_label, 51, 5
                )

        if var_key == "pd_ids":
            window = 51
            # lc[m_key] = []
            # kf = KFold(n_splits=len(discrete_classes), shuffle=False)
            # # for i, (train_i, test_i) in enumerate(kf.split(discrete_classes)):
            # test_i = [21,22]
            # for i, train_i in enumerate(discrete_classes[~np.isin(discrete_classes,test_i)]):
            #     z_train = z[np.isin(ids[ids != not_id], [test_i])][# [discrete_classes])][#[test_i])][
            #         ::window
            #     ]
            #     # print(discrete_classes[test_i], print(discrete_classes[train_i]))
            #     y_train = pd_label[
            #         np.isin(ids[ids != not_id], [test_i])#[discrete_classes])#[test_i])
            #     ][::window].ravel()
            #     z_test = z[np.isin(ids[ids != not_id], train_i)][#])][#discrete_classes[train_i])][
            #         ::window
            #     ]
            #     # y_test = pd_label[np.isin(ids[ids!=not_id], discrete_classes[train_i])][::window]
            #     y_test = np.zeros((len(z_test), 2))

            #     y_test[
            #         np.arange(len(z_test)),
            #         pd_label[np.isin(ids[ids != not_id], train_i)][#discrete_classes[train_i])][
            #             ::window
            #         ],
            #     ] = 1

            #     clf = LogisticRegression(solver="sag", max_iter=300, C=0.5).fit(
            #         z_train, y_train
            #     )
            #     lc[m_key] += [roc_auc_score(y_test, clf.predict_proba(z_test))]

            # mmd[m_key] = []
            # mmd_id_pd[m_key] = []
            # mmd_id_healthy[m_key] = []
            # for i in range(len(discrete_classes)):
            #     pd_i = ((fluorescence < 0.9) & (ids[ids!=not_id] == discrete_classes[i])).ravel()
            #     healthy_i = ((fluorescence >= 0.9) & (
            #         ids[ids!=not_id] == discrete_classes[i]
            #     )).ravel()

            #     pd_z_i = z[pd_i, ...]
            #     healthy_z_i = z[healthy_i, ...]
            #     mmd[m_key] += [mmd_estimate(pd_z_i[::window], healthy_z_i[::window])]

            #     for j in range(i+1, len(discrete_classes)):
            #         pd_j = ((fluorescence < 0.9) & (ids[ids!=not_id] == discrete_classes[j])).ravel()
            #         healthy_j = ((fluorescence >= 0.9) & (
            #             ids[ids!=not_id] == discrete_classes[j]
            #         )).ravel()
            #         pd_z_j = z[pd_j, ...]
            #         healthy_z_j = z[healthy_j, ...]

            #         mmd_id_pd[m_key] += [mmd_estimate(pd_z_i[::window], pd_z_j[::window])]
            #         mmd_id_healthy[m_key] += [mmd_estimate(healthy_z_i[::window], healthy_z_j[::window])]

            mmd[m_key] = []
            mmd_id_pd[m_key] = []
            mmd_id_healthy[m_key] = []
            mmd_id_pd_mat = np.zeros((len(discrete_classes), len(discrete_classes)))
            mmd_id_healthy_mat = np.zeros(
                (len(discrete_classes), len(discrete_classes))
            )
            for i in range(len(discrete_classes)):
                pd_i = (
                    (fluorescence < 0.9) & (ids[ids != not_id] == discrete_classes[i])
                ).ravel()
                healthy_i = (
                    (fluorescence >= 0.9) & (ids[ids != not_id] == discrete_classes[i])
                ).ravel()

                pd_z_i = z[pd_i, ...]
                healthy_z_i = z[healthy_i, ...]
                mmd[m_key] += [mmd_estimate(pd_z_i[::window], healthy_z_i[::window])]

                for j in range(i + 1, len(discrete_classes)):
                    pd_j = (
                        (fluorescence < 0.9)
                        & (ids[ids != not_id] == discrete_classes[j])
                    ).ravel()
                    healthy_j = (
                        (fluorescence >= 0.9)
                        & (ids[ids != not_id] == discrete_classes[j])
                    ).ravel()
                    pd_z_j = z[pd_j, ...]
                    healthy_z_j = z[healthy_j, ...]

                    mmd_id_pd_mat[i, j] = mmd_estimate(
                        pd_z_i[::window], pd_z_j[::window]
                    )
                    mmd_id_healthy_mat[i, j] = mmd_estimate(
                        healthy_z_i[::window], healthy_z_j[::window]
                    )

            mmd_id_pd_mat += np.triu(mmd_id_pd_mat, k=1).T
            mmd_id_healthy_mat += np.triu(mmd_id_healthy_mat, k=1).T

            for i in range(len(discrete_classes)):
                mmd_id_pd[m_key] += [
                    mmd[m_key][i]
                    / (
                        np.mean(mmd_id_pd_mat[i, np.arange(len(discrete_classes)) != i]))
                ]
                mmd_id_healthy[m_key] += [
                    mmd[m_key][i]
                    / (
                        np.mean(
                            mmd_id_healthy_mat[i, np.arange(len(discrete_classes)) != i]
                        )
                    )
                ]

    if var_key == "pd_speed":
        lc["Speed Only"] = log_class_rand_cv(speed, pd_label, 51, 5)

    w = 0.25  # bar width
    if var_key == "pd_speed":
        gs = subf[var_ind].add_gridspec(1, 12)
        ### Plot 5 Fold R2 Decoding
        bar_ax = subf[var_ind].add_subplot(gs[:7])
        bar_ax.set_title(
            "Logistic PD Classification with Average Speed Scrubbing".format(
                titles[var_key]
            ),
            fontsize=14,
        )
        x = np.arange(len(lc.keys()))  # x-coordinates of your bars
        # colors = [(0, 0, 1, 1), (1, 0, 0, 1)]    # corresponding colors

        bar_ax.bar(
            x,
            height=[np.mean(lc[k]) for k in lc.keys()],
            width=w,  # bar width
            # tick_label=list(lc.keys()),
            color="#9871bb",
            label="Logistic",
        )

        ### PD Decoding
        for i, key in enumerate(lc.keys()):
            # distribute scatter randomly across whole width of bar
            bar_ax.scatter(
                x[i] + np.random.uniform(-0.1, 0.1, len(lc[key])),
                lc[key],
                marker="o",
                c="k",
                s=1,
            )

        bar_ax.set_xticks(x)
        bar_ax.set_xticklabels(list(lc.keys()))
        bar_ax.set_ylabel("AUROC")

        print("AUROC Values")
        print({k: np.mean(lc[k]) for k in lc.keys()})
        # bar_ax.legend()

        bar_ax = subf[var_ind].add_subplot(gs[7:])
        x = np.arange(len(lin.keys()))
        ### Speed Decoding
        bar_ax.bar(
            x,
            height=[np.mean(lin[k]) for k in lin.keys()],
            width=w,  # bar width
            # tick_label=list(lc.keys()),
            color="#009392",
            label="Linear",
        )

        print("Linear Regression Values")
        print({k: np.mean(lin[k]) for k in lin.keys()})

        for i, key in enumerate(lin.keys()):
            # distribute scatter randomly across whole width of bar
            bar_ax.scatter(
                x[i] + np.random.uniform(-0.1, 0.1, len(lin[key])),
                lin[key],
                marker="o",
                c="k",
                s=1,
            )

        bar_ax.set_title("Linear Regression of Average Speed", fontsize=15)
        bar_ax.set_xticks(np.arange(len(lin.keys())))#+ 0.33 / 2)
        bar_ax.set_xticklabels(list(lin.keys()))
        bar_ax.set_ylabel(r"$R^2$")
        # bar_ax.legend()

    if var_key == "pd_ids":
        x = np.arange(len(mmd.keys()))
        print("Mean MMD PD v Healthy")
        print({k: np.mean(mmd[k]) for k in mmd.keys()})
        print("STD MMD PD v Healthy")
        print({k: np.std(mmd[k]) for k in mmd.keys()})

            # ax2 = f.add_subplot(gs[var_ind+1,4:])
        bar_ax = subf[var_ind].add_subplot(gs[6:])
        subf[var_ind].suptitle("Ratio of Disease MMD to Animal Identity MMD", fontsize=15)
        x = np.arange(len(mmd_id_pd.keys()))
        ### Speed Decoding
        bar_ax.bar(
            x,
            height=[np.mean(mmd_id_pd[k]) for k in mmd_id_pd.keys()],
            width=w,  # bar width
            # tick_label=list(lc.keys()),
            color = "#DEA1D1",
        )

        for i, key in enumerate(mmd_id_pd.keys()):
            # print(mmd_id_pd)
            # distribute scatter randomly across whole width of bar
            bar_ax.scatter(
                x[i] + np.random.uniform(-0.1, 0.1, len(mmd_id_pd[key])),
                mmd_id_pd[key],
                marker="o",
                c="k",
                s=1,
            )

        bar_ax.set_xticks(x)
        bar_ax.set_xticklabels(list(mmd_id_pd.keys()))
        bar_ax.set_ylabel("Maximum Mean Discrepancy")
        bar_ax.set_title(
            "Healthy Sessions",
        )

        bar_ax = subf[var_ind].add_subplot(gs[:6])
        x = np.arange(len(mmd_id_healthy.keys()))
        ### Speed Decoding
        bar_ax.bar(
            x,
            height=[np.mean(mmd_id_healthy[k]) for k in mmd_id_healthy.keys()],
            width=w,  # bar width
            # tick_label=list(lc.keys()),
            color = "#DEA1D1",
        )

        for i, key in enumerate(mmd_id_healthy.keys()):
            # print(mmd_id_healthy)
            # distribute scatter randomly across whole width of bar
            bar_ax.scatter(
                x[i] + np.random.uniform(-0.1, 0.1, len(mmd_id_healthy[key])),
                mmd_id_healthy[key],
                marker="o",
                c="k",
                s=1,
            )

        bar_ax.set_xticks(x)
        bar_ax.set_xticklabels(list(mmd_id_healthy.keys()))
        bar_ax.set_ylabel("Maximum Mean Discrepancy")
        bar_ax.set_title(
            "PD Sessions"
        )

        gs = subf[var_ind+1].add_gridspec(1, 12)
        subf[var_ind+1].suptitle("MMD Correlation to Integrated Fluorescence",fontsize=15)
        for i, k in enumerate(mmd.keys()):
            # print(mmd)
            # import pdb; pdb.set_trace()
            ax = subf[var_ind+1].add_subplot(gs[i * 3 : i * 3 + 3])
            idx = np.unique(fluorescence, return_index=True)[1]
            ax.scatter(fluorescence[np.sort(idx)][:-1], mmd[k])
            ax.set_xlabel("Integrated Fluorescence")
            ax.set_ylabel("Maximum Mean Discrepancy")
            ax.set_title(
                "{}\nPearson: {:3f}".format(
                    k, stats.pearsonr(fluorescence[np.sort(idx)][:-1], mmd[k])[0]
                )
            )

        # bar_ax.legend()


subf[0].subplots_adjust(left=0.075,
                            bottom=0.2, 
                            right=0.96, 
                            top=0.9,
                            wspace=1.5, 
                            hspace=6)

subf[1].subplots_adjust(left=0.075,
                            bottom=0.1, 
                            right=0.96, 
                            top=0.85,
                            wspace=1.5, 
                            hspace=3)

subf[2].subplots_adjust(left=0.075,
                            bottom=0.13, 
                            right=0.96, 
                            top=0.81,
                            wspace=1.75, 
                            hspace=3)


# f.tight_layout()
plt.savefig("./results/pd_new.png",dpi=400)
