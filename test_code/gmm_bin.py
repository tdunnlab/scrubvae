import ssumo
from neuroposelib import read
import numpy as np
from sklearn.mixture import GaussianMixture
import matplotlib.pyplot as plt
from pathlib import Path
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from ssumo.data.dataset import fwd_kin_cont6d_torch
import tqdm
import torch

RESULTS_PATH = "/mnt/ceph/users/jwu10/results/vae/"


speed_bin = [1, 1.5]
analysis_key = "mi_64_bw/double_checking"
epoch = 270
path = RESULTS_PATH + analysis_key + "/"
config = read.config(path + "model_config.yaml")
connectivity = read.connectivity_config(config["data"]["skeleton_path"])
vis_path = path + "/vis_latents/"
Path(vis_path).mkdir(parents=True, exist_ok=True)

loader = ssumo.get.mouse_data(
    data_config=config["data"],
    window=config["model"]["window"],
    train=True,
    data_keys=[
        "avg_speed_3d",
        "heading",
        "x6d",
        "root",
        "offsets",
        # "raw_pose",
    ],
    shuffle=False,
    normalize=["avg_speed_3d"],
)
n_keypts = loader.dataset.n_keypts
kinematic_tree = loader.dataset.kinematic_tree

speed = loader.dataset[:]["avg_speed_3d"].cpu().detach().numpy()
heading = loader.dataset[:]["heading"].cpu().detach().numpy()
heading = np.arctan2(heading[:, 0], heading[:, 1])
avg_speed = speed.mean(-1)
# z = StandardScaler().fit_transform(z)

print(config["data"]["direction_process"])
model = ssumo.get.model(
    model_config=config["model"],
    load_model=config["out_path"],
    epoch=epoch,
    disentangle_config=config["disentangle"],
    n_keypts=n_keypts,
    direction_process=config["data"]["direction_process"],
    loss_config=config["loss"],
    arena_size=loader.dataset.arena_size,
    kinematic_tree=kinematic_tree,
    bound=config["data"]["normalize"] is not None,
    discrete_classes=loader.dataset.discrete_classes,
    verbose=1,
)
model.eval()

z = ssumo.get.latents(
    config, model, epoch, loader, device="cuda", dataset_label="Train"
)
assert len(z) == len(speed)
assert len(z) == len(loader.dataset)

pca = PCA().fit(z)
print("Explained Variance")
print(np.cumsum(pca.explained_variance_ratio_))
exp_var_99 = np.where(np.cumsum(pca.explained_variance_ratio_) > 0.95)[0][1]
# z = pca.transform(z)[:,:exp_var_99]

loader.dataset.data["z"] = torch.cat(
    [z, torch.from_numpy(speed)], dim=-1
)
pose = []

for batch_idx, data in enumerate(tqdm.tqdm(loader)):
    x_hat = model.decoder(data["z"].cuda()).moveaxis(-1, 1)
    x6d = x_hat[..., :-3].reshape(data["z"].shape[0], model.window, -1, 6)
    offsets = data["offsets"].cuda()

    root = model.inv_normalize_root(x_hat[..., -3:]).reshape(
        data["z"].shape[0] * model.window, 3
    )

    pose_batch = fwd_kin_cont6d_torch(
        x6d.reshape((-1, n_keypts, 6)),
        kinematic_tree,
        offsets.reshape((-1,) + offsets.shape[-2:]),
        root_pos=root,
        do_root_R=True,
        eps=1e-8,
    ).reshape((-1, model.window, n_keypts, 3))

    pose += [pose_batch.detach().cpu()]

pose = torch.cat(pose, dim=0).numpy()

for method in ["clusters", "speed_bin", "speed_cat"]:
    Path("{}/vis_{}_{}/".format(vis_path, method, epoch)).mkdir(
        parents=True, exist_ok=True
    )
    if method == "speed_bin":
        speed_bin_i = np.where((avg_speed > speed_bin[0]) & (avg_speed < speed_bin[1]))[
            0
        ]
    else:
        speed_bin_i = np.arange(len(speed), dtype=int)

    if method == "speed_cat":
        z_temp = np.concatenate([z, speed], axis=-1)
    else:
        z_temp = z

    k_pred = GaussianMixture(
        n_components=25,
        covariance_type="diag",
        max_iter=150,
        init_params="k-means++",
        reg_covar=0.001,
        verbose=1,
    ).fit_predict(z_temp[speed_bin_i, :])

    print(
        np.histogram(
            k_pred, bins=len(np.unique(k_pred)), range=(-0.5, np.max(k_pred) + 0.5)
        )[0]
    )

    ssumo.plot.feature_ridge(
        feature=heading[speed_bin_i],
        labels=k_pred,
        xlabel="Heading",
        ylabel="Cluster",
        x_lim=(-np.pi - 0.1, np.pi + 0.1),
        n_bins=200,
        binrange=(-np.pi, np.pi),
        path="{}/vis_{}_{}/".format(vis_path, method, epoch),
    )

    ssumo.plot.sample_clusters(
        pose[speed_bin_i, ...],
        k_pred,
        connectivity,
        "{}/vis_{}_{}/".format(vis_path, method, epoch),
    )

    f = plt.figure(figsize=(10, 5))
    plt.hist(k_pred, bins=len(np.unique(k_pred)), range=(-0.5, 25 - 0.5))
    plt.xlabel("GMM Cluster")
    plt.ylabel("# Actions")
    plt.savefig("{}/vis_{}_{}/gmm_hist.png".format(vis_path, method, epoch))
    plt.close()
