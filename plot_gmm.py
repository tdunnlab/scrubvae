import numpy as np
import matplotlib.pyplot as plt
from dappy import read
import utils
import pickle
from sklearn.decomposition import PCA

path = "/mnt/ceph/users/jwu10/results/vae/gr_scratch/avgspd_ndgre20_rc_w51_b1_midfwd_full_a1/"
config = read.config(path + "/model_config.yaml")
config["load_model"] = config["out_path"]
config["load_epoch"] = 200
load_spd_epoch = config["load_epoch"]

latents = np.load(
    "{}/latents/Train_{}.npy".format(config["out_path"], config["load_epoch"])
)
pca = PCA(n_components=10)
pca.fit(latents)
f = plt.figure(figsize=(10,10))
plt.plot(np.arange(10), np.cumsum(pca.explained_variance_ratio_))
plt.xlabel("Principal Component")
plt.ylabel("Cumulative Explained Variance Ratio")
plt.savefig(path + "cumul_pca_explained_var.png")

z_loglike = np.load("{}gmm_sphere/test_loglike_z.npy".format(path))
zsub_loglike = np.load("{}gmm_sphere/test_loglike_zsub.npy".format(path))

k = np.arange(5, 201, 5)

try:
    z_place = np.where(z_loglike==0)[0][0]
    zsub_place = np.where(zsub_loglike==0)[0][0]
    place = min(z_place, zsub_place) # Find current iteration of gmm model
except:
    place = len(k)

# 
z_loglike = z_loglike[:place,:].mean(axis=-1)
zsub_loglike = zsub_loglike[:place,:].mean(axis=-1)

# Plot log likelihoods of clustering in z vs orthogonal z subspace
f = plt.figure(figsize=(10,10))
plt.plot(k[:place], z_loglike, label="$z$")
plt.plot(k[:place], zsub_loglike, label=r"$\tilde{z}$")
plt.xlabel("# of Clusters")
plt.ylabel("Test Log Likelihood")
plt.legend()
plt.savefig(path+"z_ll_sphere.png")
import pdb; pdb.set_trace()

vae, spd_decoder, device = utils.init_model(config, 18, config["invariant"])

def KL(m1, c1, m2, c2):
    diff = (m2 - m1)[:, None]
    kl = np.log(np.linalg.det(c2)) - np.log(np.linalg.det(c1))
    kl -= len(m1)
    kl += diff.T@np.linalg.inv(c2)@diff
    kl += np.diag(np.linalg.inv(c2)@c1).sum()
    import pdb; pdb.set_trace()
    return 0.5*kl

for i in range(place):
    for j in range(5):
        z_gmm = pickle.load(open("{}gmm_pca3/z_gmm_k{}_{}_gmm.p".format(path,k[i],j),"rb"))
        z_gmm_covs = np.identity(z_gmm.covariances_.shape[-1])[None,...].repeat(k[i],axis=0)
        z_gmm_covs = z_gmm_covs * z_gmm.covariances_[...,None]
        spd_weights = spd_decoder.weight.cpu().detach().numpy()
        import pdb; pdb.set_trace()
        spd_weights = pca.transform(spd_weights)
        nrm = (spd_weights @ spd_weights.T).ravel()

        I_wwt = np.identity(len(spd_weights)) - (spd_weights.T @ spd_weights)/nrm
        assert ((I_wwt @ z_gmm_covs)[0] - (I_wwt@z_gmm_covs[0])).sum() == 0
        zsub_cov = (I_wwt @ z_gmm_covs)@(I_wwt)
        zsub_means = (I_wwt @ z_gmm.means_.T).T

        # print([np.linalg.det(mat) for mat in z_gmm.covariances_])
        import pdb; pdb.set_trace()

        ## Option 1 where we calculate JS divergence btwn clusters in subspaces
        # import pdb; pdb.set_trace()
        for row, col in zip(*np.triu_indices(k[i], 1)):
            mu1, cov1 = zsub_means[row], zsub_cov[row]
            mu2, cov2 = zsub_means[col], zsub_cov[col]

            kl12 = KL(mu1, cov1, mu2, cov2)
            kl21 = KL(mu2, cov2, mu1, cov1)

            js = (kl12 + kl21)/2

            import pdb; pdb.set_trace()

