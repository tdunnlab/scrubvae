import numpy as np
import matplotlib.pyplot as plt
import ssumo
import pickle

# from base_path import RESULTS_PATH, CODE_PATH
import sys
from pathlib import Path
from dappy import read
from sklearn.decomposition import PCA
import colorcet as cc
from scipy.stats import circvar
from mpl_toolkits.axes_grid1 import make_axes_locatable

plt.rcParams["font.family"] = "sans-serif"
plt.rcParams["font.sans-serif"] = "Arial"
plt.rcParams["font.size"] = "10"

CODE_PATH = "/mnt/home/jwu10/working/ssumo/"
RESULTS_PATH = "/mnt/ceph/users/jwu10/results/vae/"

titles = {
    "heading": "Heading Direction",
    "avg_speed_3d": "Average Speed",
    "ids": "Animal ID",
}

f = plt.figure(figsize=(15, 7))
subf = f.subfigures(3, 1)
for var_ind, var_key in enumerate(["avg_speed_3d", "heading", "ids"]):
    models = read.config(CODE_PATH + "configs/exp_finals.yaml")[var_key]
    models = {m[0]: [m[1], m[2]] for m in models}
    print(models)

    if var_ind == 0:
        config = read.config(
            RESULTS_PATH + models["Conditional VAE"][0] + "/model_config.yaml"
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
        cmap = cc.cm["colorwheel"]
    elif var_key == "avg_speed_3d":
        var_plt = var.mean(axis=-1)
        cmap = "viridis"
    elif var_key == "ids":
        var_plt = var.ravel().astype(int)
        permute_ind = np.random.permutation(len(var_plt[::downsample]))
        cmap = ["#EF476F", "#FFD166", "#06D6A0"]# ssumo.plot.constants.PALETTE_2
#["#E9002D", "#FFAA00", "#00B000"]
    i = 0
    ax = subf[var_ind].subplots(1, len(models.keys())+2)
    ghost_ax_gs = subf[var_ind].add_gridspec(1, len(models.keys())+2)
    subf[var_ind].suptitle("{} Scrubbing".format(titles[var_key]), fontsize=14)
    for model in models.keys():
        path = "{}{}/".format(RESULTS_PATH, models[model][0])
        print("Loading from {}".format(path))
        z = np.load(path + "latents/Train_{}.npy".format(models[model][1]))

        pca = PCA(
            n_components=2 if model not in ["Vanilla VAE", "Conditional VAE"] else 4
        )
        pcs = pca.fit_transform(z[::downsample, :].astype(np.float64))

        # ax = subf[var_ind].add_subplot(gs[i])

        if var_key == "ids":
            im = ax[i].scatter(
                pcs[permute_ind, 0], pcs[permute_ind, 1], s=0.5, c=np.array(cmap)[var_plt[::downsample]][permute_ind], alpha=0.5
            )
        else:
            im = ax[i].scatter(pcs[:, 0], pcs[:, 1], s=0.5, c=var_plt[::downsample], cmap=cmap)

        ax[i].set_xlabel("PC 1")
        ax[i].set_ylabel("PC 2")
        ax[i].get_xaxis().set_ticks([])
        ax[i].get_yaxis().set_ticks([])
        ax[i].spines['top'].set_visible(False)
        ax[i].spines['right'].set_visible(False)
        ax[i].spines['bottom'].set_visible(False)
        ax[i].spines['left'].set_visible(False)
        if model not in ["Vanilla VAE", "Conditional VAE"]:
            ax[i].set_title(model)
        i += 1

        if model in ["Vanilla VAE", "Conditional VAE"]:
            # ax = f.add_subplot(gs[var_ind, i])
            if var_key == "ids":
                im = ax[i].scatter(
                    pcs[permute_ind, 2], pcs[permute_ind, 3], s=0.5, c=np.array(cmap)[var_plt[::downsample]][permute_ind], alpha=0.5
                )
            else:
                im = ax[i].scatter(
                    pcs[:, 2], pcs[:, 3], s=0.5, c=var_plt[::downsample], cmap=cmap
                )

            ax[i].spines['top'].set_visible(False)
            ax[i].spines['right'].set_visible(False)
            ax[i].spines['bottom'].set_visible(False)
            ax[i].spines['left'].set_visible(False)
            ax[i].set_xlabel("PC 3")
            ax[i].set_ylabel("PC 4")
            ax[i].get_xaxis().set_ticks([])
            ax[i].get_yaxis().set_ticks([])
            ghost_ax = subf[var_ind].add_subplot(ghost_ax_gs[i-1:i+1])
            ghost_ax.axis("off")
            ghost_ax.set_title(model)
            # ax[i].set_title(model)
            # title_ax = f.add_subplot(gs[i-1:i+1])
            # title_ax.axis("off")
            # title_ax.set_title(model)
            i += 1
        # else:
        #     ax[i].set_title(model)

    # if var_key != "ids":
        # ghost_ax = f.add_subplot(gs[-1])
        # ghost_ax.axis("off")
        # divider = make_axes_locatable(ax)
        # cax = divider.append_axes('right', size='5%', pad=0.05)
    if var_key == "ids":
        import matplotlib.patches as mpatches
        patch = []
        for ind, color in enumerate(cmap):
            patch += [mpatches.Patch(color=color, label="Animal {}".format(ind+1))]
        subf[var_ind].legend(handles = patch, loc="lower center", ncol=8)
            
        # handles, labels = ax[i-1].get_legend_handles_labels()
        # subf[var_ind].legend(handles, labels, loc="lower center", ncol=8)

    if var_key == "heading":
        # cb = ax[-1].cax.colorbar(im)
        # cb.set_label_text("Heading Angle (rad)")
        cax = subf[var_ind].add_axes((0.95, 0.1, 0.01, 0.75))
        subf[var_ind].colorbar(im, cax=cax, label="Heading Angle (rad)")
    elif var_key == "avg_speed_3d":
        cax = subf[var_ind].add_axes((0.95, 0.1, 0.01,0.75))
        # cb = ax[-1].cax.colorbar(im)
        # cb.set_label_text("Average Speed")
        subf[var_ind].colorbar(im, cax=cax, label="Average Speed")

    if var_key == "ids":
        subf[var_ind].subplots_adjust(left=0.03,
                            bottom=0.25, 
                            right=0.97, 
                            top=0.7,
                            wspace=0.75, 
                            hspace=4)
    else:
        subf[var_ind].subplots_adjust(left=0.03,
                            bottom=0.13, 
                            right=0.93, 
                            top=0.75,
                            wspace=0.75, 
                            hspace=4)
# f.tight_layout()
plt.savefig("./results/pca_final.png")
