import numpy as np
import matplotlib.pyplot as plt
import ssumo
import pickle
# from base_path import RESULTS_PATH, CODE_PATH
import sys
from pathlib import Path
from neuroposelib import read
from sklearn.decomposition import PCA
import colorcet as cc
from scipy.stats import circvar
CODE_PATH = "/mnt/home/jwu10/working/ssumo/"
RESULTS_PATH = "/mnt/ceph/users/jwu10/results/vae/"

var_key = "avg_speed_3d"
titles = {"heading": "Heading Direction",
          "avg_speed_3d": "Average Speed",
          "ids": "Animal ID"}

models = read.config(CODE_PATH + "configs/exp_{}.yaml".format(var_key))["models"]
models = {m[0]:[m[1], m[2]] for m in models}
mlp_cv, lin_cv, gmm_var = {},{},{}

config = read.config(
    RESULTS_PATH + models["Conditional VAE"][0] + "/model_config.yaml"
)
loader = ssumo.get.mouse_data(
    data_config=config["data"],
    window=config["model"]["window"],
    train=True,
    data_keys=[var_key],
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
    var_plt = var
    cmap = ssumo.plot.constants.PALETTE_2

f = plt.figure(figsize=(30, 10))
gs = f.add_gridspec(2, 5)
ax_gmm_ll = f.add_subplot(gs[0,-1])
for i, model in enumerate(models.keys()):
    ax = f.add_subplot(gs[1,i])
    path = "{}{}/".format(RESULTS_PATH, models[model][0])
    
    model_lin_cv = pickle.load(open(path + "linear_cv_reg_Train.p", "rb"))
    lin_cv[model] = np.array(model_lin_cv[var_key]["R2"])[np.array(model_lin_cv["epochs"])==models[model][1]].ravel()

    model_mlp_cv = pickle.load(open(path + "mlp_cv_reg_Train.p", "rb"))
    mlp_cv[model] = np.array(model_mlp_cv[var_key]["R2"])[np.array(model_mlp_cv["epochs"])==models[model][1]].ravel()

    z = np.load(path + "latents/Train_{}.npy".format(models[model][1]))

    k_pred, gmm = ssumo.eval.cluster.gmm(
        latents=z,
        n_components=25,
        label="z_{}".format(models[model][1]),
        path=path,
        covariance_type="diag",
    )
    gmm_var[model] = []
    for i in range(25):
        if var_key == "heading":
            gmm_var[model] += [[circvar(var_plt[k_pred == i], high=np.pi, low=-np.pi),(k_pred==i).sum()/len(k_pred)*25]]
        elif var_key == "avg_speed_3d":
            gmm_var[model] += [np.var(var_plt)*(k_pred==i).sum()/len(k_pred)*25]
        elif var_key == "ids":
            hist = np.histogram(var_plt[k_pred==i],bins=np.arange(var_plt.max() + 1)-0.5)/(k_pred==i).sum()
            entropy = hist * np.log2(1/hist)
            gmm_var[model] += [entropy.sum()]

    pca = PCA(n_components=2)
    pcs = pca.fit_transform(
        z[::downsample, :].astype(np.float64)
    )

    im = ax.scatter(pcs[:, 0], pcs[:, 1], s=10, c=var_plt[::downsample], cmap = cmap)
    ax.set_title(
        model
        # "Variance Explained: {:3f}".format(
        #     np.cumsum(pca.explained_variance_ratio_)[-1]
        # )
    )

    ax.set_xlabel("PC 1")
    ax.set_ylabel("PC 2")
    ax.get_xaxis().set_ticks([])
    ax.get_yaxis().set_ticks([])

    try:
        gmm_ll = np.load("{}gmm_ll/gmm_ll_{}.npy".format(path, models[model][1]))
        k = np.arange(5, 101, 5)
        try:
            place = np.where(gmm_ll==0)[0][0]
        except:
            place = len(k)
        # 
        gmm_ll = gmm_ll[:place,:].mean(axis=-1)
        # Plot log likelihoods of clustering in z
        ax_gmm_ll.plot(k[:place], gmm_ll, label=model)
        ax_gmm_ll.set_xlabel("# of Clusters")
        ax_gmm_ll.set_ylabel("Test Log Likelihood")
        ax_gmm_ll.legend()
    except:
        continue

if var_key == "heading":
    f.colorbar(im, ax=ax, label = "Heading Angle (rad)")
elif var_key == "avg_speed_3d":
    f.colorbar(im, ax=ax, label = "Average Speed")
### Plot 5 Fold R2 Decoding
bar_ax = f.add_subplot(gs[0,:2])
bar_ax.set_title("5-Fold {} Decoding from Latents".format(titles[var_key]))
w = 0.25    # bar width
x = np.arange(len(models.keys())) + 0.5 # x-coordinates of your bars
# colors = [(0, 0, 1, 1), (1, 0, 0, 1)]    # corresponding colors
bar_ax.bar(x,
       height=[np.mean(lin_cv[k]) for k in models.keys()],
       width=w,    # bar width
       tick_label=list(lin_cv.keys()),
       label="Linear"
       )
for i, key in enumerate(lin_cv.keys()):
    # distribute scatter randomly across whole width of bar
    bar_ax.scatter(x[i] + np.zeros(len(lin_cv[key])), lin_cv[key], marker = 'o', c="k")

bar_ax.bar(x+0.33,
       height=[np.mean(mlp_cv[k]) for k in models.keys()],
       width=w,    # bar width
       tick_label=list(mlp_cv.keys()),
       label="MLP"
       )
for i, key in enumerate(mlp_cv.keys()):
    # distribute scatter randomly across whole width of bar
    bar_ax.scatter(x[i] +0.33 + np.zeros(len(mlp_cv[key])), mlp_cv[key], marker = 'o', c="k")
bar_ax.set_ylabel(r"$\mathregular{R^2}$")
bar_ax.legend()

# import pdb; pdb.set_trace()
var_ax = f.add_subplot(gs[0,2:4])
var_ax.set_title("{} GMM Cluster Variance".format(titles[var_key]))
# colors = [(0, 0, 1, 1), (1, 0, 0, 1)]    # corresponding colors
var_ax.bar(x+0.5,
       height=[np.mean(np.prod(gmm_var[k],axis=-1)) for k in models.keys()],
    #    yerr=[np.std(yi) for yi in y],    # error bars
    #    capsize=12, # error bar cap width in points
       width=w,    # bar width
       tick_label=list(gmm_var.keys()),
    #    color=(0,0,0,0),  # face color transparent
    #    edgecolor=colors,
       #ecolor=colors,    # error bar colors; setting this raises an error for whatever reason.
       )
for i, key in enumerate(gmm_var.keys()):
    # distribute scatter randomly across whole width of bar
    var_ax.scatter(x[i]+0.5 + np.zeros(len(gmm_var[key])), np.array(gmm_var[key])[:,0], marker = 'o', c="k")
var_ax.set_ylabel(r"$\sigma^2$")

f.tight_layout()
plt.savefig("./results/{}.png".format(var_key))