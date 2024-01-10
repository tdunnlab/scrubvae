import numpy as np
import utils
from dappy import read
from data.dataset import MouseDataset
import matplotlib.pyplot as plt
from sklearn.metrics import r2_score
import torch
import torch.nn as nn
from tqdm import trange
import torch.optim as optim
from sklearn.linear_model import LinearRegression
from sklearn.decomposition import PCA

out_base = "/mnt/home/jwu10/working/ceph/results/vae/"

path = "avgspd_ndgre20_rc_w51_b1_midfwd_full_a1"
base_path = "/mnt/ceph/users/jwu10/results/vae/gr_scratch/"
config = read.config(base_path + path + "/model_config.yaml")
config["load_model"] = config["out_path"]
config["load_epoch"] = 650

dataset = MouseDataset(
    data_path=config["data_path"],
    skeleton_path="/mnt/home/jwu10/working/behavior_vae/configs/mouse_skeleton.yaml",
    train=True,
    window=config["window"],
    stride=1,
    direction_process=config["direction_process"],
    get_speed=config["speed_decoder"],
    arena_size=config["arena_size"],
    invariant=config["invariant"],
    get_raw_pose=False,
)

spd_true = dataset[:]["speed"].mean(dim=-1, keepdim=True).detach().cpu().numpy()

vae, spd_decoder, device = utils.init_model(config, 18, config["invariant"])
vae.eval()
spd_decoder.eval()
latents = [np.load("{}/rc_w51_midfwd_full/latents/Train_600.npy".format(out_base))]
latents += [utils.get_latents(vae, dataset, config, device, "Train")]

if config["gradient_reversal"]:
    spd_weights = spd_decoder.decoder.weight.cpu().detach().numpy()
else:
    spd_weights = spd_decoder.weight.cpu().detach().numpy()

nrm = (spd_weights @ spd_weights.T).ravel()
avg_spd_o = latents[1] @ spd_weights.T
latents += [latents[1] - (avg_spd_o @ spd_weights) / nrm]

f, ax = plt.subplots(1,2, figsize=(7,3), dpi=400)
titles = ["Vanilla VAE", "Semi-Supervised VAE", "Speed Invariant Subspace"]
for i, z in enumerate(latents[:2]):
    # pca = PCA(n_components=20).fit(z)
    # cum_exp_var = np.cumsum(pca.explained_variance_ratio_)
    # print(cum_exp_var)
    # print(np.where(cum_exp_var>0.97*cum_exp_var.max())[0][0]+1)
    # z_trans = pca.transform(z)[:, :np.where(cum_exp_var>0.9*cum_exp_var.max())[0][0]+1]
    spd_pred = LinearRegression().fit(z, spd_true).predict(z)
    # if i == 0:
    # spd_pred = LinearRegression().fit(z, spd_true).predict(z)
    # z_b = np.append(z, np.ones((len(z),1)),axis=-1)
    # w_s = (np.linalg.inv(z_b.T @ z_b) @ z_b.T) @ spd_true
    # spd_pred = z_b @ w_s
    # if i == 2:
    #     import pdb; pdb.set_trace()
    #     spd_decoder.reversal.weight = torch.tensor(w_s[:-1])
    #     spd_decoder.reversal.bias = torch.tensor(w_s[-1])
    #     spd_pred = spd_decoder(latents[1].cuda())[i-1].cpu().detach().numpy()
    # else:
    #     spd_pred = spd_decoder(latents[1].cuda())[i-1].cpu().detach().numpy()
    ax[i].scatter(spd_pred, spd_true, alpha=0.001, s=1, marker=".")

    y_e_x = np.linspace(0,5,100)
    ax[i].plot(y_e_x, y_e_x, "k-", label="y=x")
    ax[i].legend()
    # ax[i].axis("equal")
    ax[i].set_ylim(bottom=0,top=5)
    ax[i].set_xlabel("Decoder Predicted Speed")
    ax[i].set_ylabel("Target Speed")
    ax[i].set_title(titles[i])

    print(r2_score(spd_true, spd_pred))
    # import pdb; pdb.set_trace()

f.tight_layout()
plt.savefig(config["out_path"] + "/spd_pred.png")



# # Look at decodability of random subspaces
# print("Random Subspaces")
# for i in range(20):
#     rand_weights = (2*(np.random.rand(128) - 0.5))[None,:]
#     nrm = (rand_weights @ rand_weights.T).ravel()
#     avg_spd_o = latents[0] @ rand_weights.T
#     rand_subspace = latents[0] - (avg_spd_o @ rand_weights) / nrm

#     # z_b = np.append(rand_subspace, np.ones((len(rand_subspace),1)),axis=-1)
#     # w_s = (np.linalg.inv(z_b.T @ z_b) @ z_b.T) @ spd_true
#     # spd_pred = z_b @ w_s

#     spd_pred = LinearRegression().fit(rand_subspace, spd_true).predict(rand_subspace)

#     print(r2_score(spd_true, spd_pred))
#     # if r2_score(spd_true, spd_pred)<-45:
#     #     f = plt.figure()
#     #     plt.scatter(spd_pred, spd_true, alpha=0.01, s=2, marker=".")
#     #     y_e_x = np.linspace(0,5,100)
#     #     plt.plot(y_e_x, y_e_x, "k-", label="y=x")
#     #     plt.legend()
#     #     plt.xlabel("Decoder Predicted Speed")
#     #     plt.ylabel("Target Speed")
#     #     plt.title("R^2: {:.2f}".format(r2_score(spd_true, spd_pred)))
#     #     plt.savefig(config["out_path"] + "/spd_pred_50.png")