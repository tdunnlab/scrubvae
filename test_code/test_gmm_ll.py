import utils
import numpy as np
import torch
from pathlib import Path
from dappy import read
import tqdm
from sklearn.decomposition import PCA
import scipy.linalg as spl
from data.dataset import MouseDataset
import sys

z_type = sys.argv[1]
### Set/Load Parameters
path = "avgspd_ndgre1_rc_w51_b1_midfwd_full_a05"
base_path = "/mnt/ceph/users/jwu10/results/vae/gr_scratch/"
config = read.config(base_path + path + "/model_config.yaml")
config["load_model"] = config["out_path"]
config["load_epoch"] = 470
gmm_path = config["out_path"] + "/gmm_full/"
Path(gmm_path).mkdir(parents=True, exist_ok=True)

# ### Load Dataset
# dataset = MouseDataset(
#     data_path=config["data_path"],
#     skeleton_path="./configs/mouse_skeleton.yaml",
#     train=True,
#     window=config["window"],
#     stride=config["stride"],
#     direction_process=config["direction_process"],
#     get_speed=config["speed_decoder"] is not None,
#     get_raw_pose=False,
#     get_root=config["arena_size"] is not None,
#     arena_size=config["arena_size"],
#     conditional=config["conditional"],
# )

latents = np.load(
    "{}/latents/Train_{}.npy".format(config["out_path"], config["load_epoch"])
)
# pca = PCA(n_components=11, svd_solver="full").fit(latents)

if z_type == "sub":
    vae, spd_decoder, device = utils.init_model(config, 18, config["conditional"])
    spd_decoder.eval()
    if "gr" in path:
        spd_weights = spd_decoder.decoder.weight.cpu().detach().numpy()
    else:
        spd_weights = spd_decoder.weight.cpu().detach().numpy()

    U_orth = spl.null_space(spd_weights)
    latents = latents @ U_orth
    # nrm = (spd_weights @ spd_weights.T).ravel()
    # avg_spd_o = latents @ spd_weights.T
    # latents = latents - (avg_spd_o @ spd_weights) / nrm
    

# latents = latents[:,np.where(latents.std(axis=0)>0.1)[0]]

# latents = pca.transform(latents)
# # (latents @ pca.components_.T) @ (pca.components_ @ spd_weights.T)
# # pca.transfrom(latents) @ pca.transform(spd_weights).T
# spd = dataset[:]["speed"].mean(-1).detach().numpy()
# wts = np.linalg.inv(latents.T @ latents)@latents.T @ spd[:,None]
# wts = wts/np.sqrt(wts.T @ wts)
# wts_trans = pca.transform(spd_weights)/np.linalg.norm(pca.transform(spd_weights))

# import pdb; pdb.set_trace()

n_splits = 4
np.random.seed(100)
permute = np.random.permutation(np.arange(len(latents)))
inds = np.arange(0, len(permute) + 1, len(permute) / n_splits).astype(int)
x_train, x_test = [], []
for i in range(n_splits):
    mask = np.ones(len(permute), dtype=bool)
    mask[inds[i] : inds[i + 1]] = False
    x_train_i = latents[permute[mask], :]
    x_test_i = latents[permute[~mask], :]
    # pca = PCA(n_components=11, svd_solver="full").fit(x_train_i)

    # if z_type == "sub":
    #     vae, spd_decoder, device = utils.init_model(config, 18, config["conditional"])
    #     spd_weights = spd_decoder.weight.cpu().detach().numpy()
    #     # spd_weights = pca.transform(spd_weights)
    #     nrm = (spd_weights @ spd_weights.T).ravel()
    #     I_wwt = np.identity(len(spd_weights.T)) - (spd_weights.T @ spd_weights) / nrm
    #     x_train_i = x_train_i @ I_wwt.T
    #     x_test_i = x_test_i @ I_wwt.T

    x_train += [x_train_i]
    x_test += [x_test_i]


k = np.arange(5, 101, 5)
test_loglike = np.zeros((len(k), n_splits))
for i in tqdm.trange(len(k)):
    print("GMM with {} components".format(k[i]))
    for j in tqdm.trange(n_splits):
        model = utils.get_gmm_clusters(
            latents = x_train[j],
            n_components = k[i],
            label="z{}_gmm_k{}_{}".format(z_type, k[i], j),
            path=gmm_path,
            covariance_type="full",
        )[1]

        test_loglike[i, j] = model.score(x_test[j])
        print(test_loglike[i, j])

        np.save("{}test_loglike_z{}.npy".format(gmm_path, z_type), test_loglike)
