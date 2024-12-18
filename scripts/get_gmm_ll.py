import scrubbed_cvae
import numpy as np
import torch
from pathlib import Path
from neuroposelib import read
import tqdm
from sklearn.decomposition import PCA
import scipy.linalg as spl
import sys
from scrubbed_cvae.eval.metrics import custom_cv_5folds

RESULTS_PATH = "/mnt/ceph/users/jwu10/results/vae/"

models = {
    "Vanilla VAE": ["vanilla/64", 300],
    # "Beta VAE": [],
    "Condtional VAE": ["mals_64/cvae_64", 280],
    "Gradient Reversal": ["gr_64/50", 290],
    "Recursive Least Squares": ["mals_64/mals_p1_20", 265],
    "Mutual Information": ["mi_64/bw75_500", 220],
}
model_keys = list(models.keys())
if sys.argv[1].isdigit():
    idx = int(sys.argv[1])
    analysis_key = models[model_keys[idx]][0]
    epoch = models[model_keys[idx]][1]
else:
    analysis_key = models[sys.argv[1]][0]
    epoch = models[sys.argv[1]][1]

config = read.config(
    RESULTS_PATH + analysis_key + "/model_config.yaml"
)

loader = scrubbed_cvae.get.mouse_data(
            data_config=config["data"],
            window=config["model"]["window"],
            train=True,
            data_keys=["ids"],
            shuffle=False,
            normalize=[],
        )
ids = loader.dataset[:]["ids"].detach().cpu().numpy().ravel()
z = np.load(
    "{}/latents/Train_{}.npy".format(config["out_path"], epoch)
)

gmm_path = config["out_path"] + "/gmm_ll/"
Path(gmm_path).mkdir(parents=True, exist_ok=True)
n_splits = 5
k = np.arange(5, 101, 5)
test_loglike = np.zeros((len(k), n_splits))
for i in tqdm.trange(len(k)):
    print("GMM with {} components".format(k[i]))
    for j in tqdm.trange(n_splits):
        idx_train, idx_test = custom_cv_5folds(j, ids)

        _, gmm = scrubbed_cvae.eval.cluster.gmm(
            latents=z[idx_train,:],
            n_components=k[i],
            label="gmm_k{}_cv{}".format(k[i], j),
            path=gmm_path,
            covariance_type="diag",
        )

        test_loglike[i, j] = gmm.score(z[idx_test,:])
        print(test_loglike[i, j])

        np.save("{}gmm_ll_{}.npy".format(gmm_path, epoch), test_loglike)
