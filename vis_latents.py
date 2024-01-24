import ssumo
from torch.utils.data import DataLoader
from dappy import read
import torch
from dappy import visualization as vis
import numpy as np
from pathlib import Path
import tqdm
import pickle

z_null = None
gen_means_cluster = False
gen_samples_cluster = False
gen_actions = False
vis_clusters = True

path = "gre1_b1_true_x360"
base_path = "/mnt/ceph/users/jwu10/results/vae/heading/"
out_path = base_path + path + "/vis_latents/"
config = read.config(base_path + path + "/model_config.yaml")
k = 25  # Number of clusters
config["model"]["load_model"] = config["out_path"]
config["model"]["start_epoch"] = 250
config["data"]["stride"] = 2
# config["speed_decoder"] = None
Path(out_path).mkdir(parents=True, exist_ok=True)

connectivity = read.connectivity_config(config["data"]["skeleton_path"])
dataset_label = "Train"
### Load Dataset
dataset = ssumo.data.get_mouse(
    data_config=config["data"],
    window=config["model"]["window"],
    train=dataset_label == "Train",
    data_keys=["x6d", "root", "offsets"] + config["disentangle"]["features"],
)
loader = DataLoader(
    dataset=dataset, batch_size=config["train"]["batch_size"], shuffle=False
)
vae, device = ssumo.model.get(
    model_config=config["model"],
    disentangle_config=config["disentangle"],
    n_keypts=dataset.n_keypts,
    direction_process=config["data"]["direction_process"],
    arena_size=dataset.arena_size,
    kinematic_tree=dataset.kinematic_tree,
    verbose=1,
)
kinematic_tree = dataset.kinematic_tree
n_keypts = dataset.n_keypts

latents = ssumo.evaluate.get.latents(vae, dataset, config, device, dataset_label)
mean_offsets = dataset.data["offsets"].mean(axis=(0, -1)).cuda()
latent_means = latents.mean(axis=0)
latent_std = latents.std(axis=0)
num_latents = latents.shape[-1]

# import pdb; pdb.set_trace()

if z_null is not None:
    dis_w = vae.disentangle[z_null].decoder.weight.cpu().detach().numpy()
    import scipy.linalg as spl
    U_orth = spl.null_space(dis_w)
    latents = latents @ U_orth
    # nrm = (spd_weights @ spd_weights.T).ravel()
    # avg_spd_o = latents @ spd_weights.T
    # latents = latents - (avg_spd_o @ spd_weights) / nrm

### Visualize clusters
if vis_clusters:
    # k_pred, gmm = ssumo.evaluate.cluster.gmm(
    #     latents=latents,
    #     n_components=k,
    #     label="z_sub",
    #     path=out_path,
    #     covariance_type="full",
    # )

    k_pred = ssumo.evaluate.cluster.dbscan( latents=latents, eps=34.4, min_samples=256, label="z", path=out_path, )

    print(np.histogram(k_pred, bins=len(np.unique(k_pred)), range=(-1.5, np.max(k_pred)+0.5))[0])
    import pdb; pdb.set_trace()
    # assert len(k_pred) == len(dataset)
    # from sklearn.cluster import DBSCAN
    # k_pred = DBSCAN().fit_predict(latents)

    ### Sample 9 videos from each cluster
    n_samples = 9
    indices = np.arange(len(k_pred))
    for cluster in np.unique(k_pred):
        label_idx = indices[k_pred == cluster]
        num_points = min(len(label_idx), n_samples)
        permuted_points = np.random.permutation(label_idx)
        sampled_points = []
        for i in range(len(permuted_points)):
            if len(sampled_points) == num_points:
                break
            elif any(np.abs(permuted_points[i] - np.array(sampled_points)) < 100):
                continue
            else:
                sampled_points += [permuted_points[i]]

        print("Plotting Poses from Cluster {}".format(cluster))
        print(sampled_points)

        num_points = len(sampled_points)

        raw_pose = ssumo.data.dataset.fwd_kin_cont6d_torch(
            dataset[sampled_points]["x6d"].reshape(-1, dataset.n_keypts, 6),
            dataset.kinematic_tree,
            dataset[sampled_points]["offsets"].reshape(-1, dataset.n_keypts, 3),
            # (torch.abs(mean_offsets)[:, None].detach().cpu() * dataset.offset).repeat( num_points * config["window"], 1, 1 ),
            root_pos=dataset[sampled_points]["root"].reshape(-1, 3),
            do_root_R=True,
        ).numpy()

        if num_points == n_samples:
            n_trans = 100
            plot_trans = (
                np.array(
                    [
                        [0, 0],
                        [1, 1],
                        [1, -1],
                        [-1, 1],
                        [-1, -1],
                        [1.5, 0],
                        [0, 1.5],
                        [-1.5, 0],
                        [0, -1.5],
                    ]
                )
                * n_trans
            )
            plot_trans = np.append(plot_trans, np.zeros(n_samples)[:, None], axis=-1)
            raw_pose += np.repeat(plot_trans, vae.window, axis=0)[:, None, :]
        # raw_pose = dataset[sampled_points]["raw_pose"].reshape(
        #     num_points * config["window"], dataset.n_keypts, 3
        # )

        vis.pose.arena3D(
            raw_pose,
            connectivity,
            frames=np.arange(num_points) * vae.window,
            centered=False,
            N_FRAMES=vae.window,
            fps=30,
            dpi=200,
            VID_NAME="cluster{}.mp4".format(cluster),
            SAVE_ROOT=out_path + "/sampled_clusters/",
        )
import pdb

pdb.set_trace()

if gen_means_cluster:
    k_pred, gmm = utils.get_gmm_clusters(latents, k, label="cluster", path=out_path)
    assert len(k_pred) == len(dataset)
    gmm_means = torch.tensor(gmm.means_, dtype=torch.float32)
    eps = torch.randn_like(gmm_means)
    gmm_L = torch.linalg.cholesky(torch.tensor(gmm.covariances_)).type(torch.float32)
    gmm_gen = torch.matmul(gmm_L, eps[..., None]).squeeze().add_(gmm_means)
    # import pdb; pdb.set_trace()
    x_o = vae.decoder(gmm_gen.cuda()).moveaxis(-1, 1)

    if config["arena_size"] is None:
        x6d_o = x_o.reshape((k * config["window"], -1, 6))
    else:
        x6d_o = x_o[..., :-3].reshape(-1, dataset.n_keypts, 6)
        root_o = inv_normalize_root(x_o[..., -3:], arena_size).reshape(-1, 3)

    pose = fwd_kin_cont6d_torch(
        x6d_o,
        dataset.kinematic_tree,
        (mean_offsets[:, None] * dataset.offset.cuda()).repeat(
            k * config["window"], 1, 1
        ),
        root_o,
        do_root_R=True,
    )

    vis.pose.grid3D(
        pose.cpu().detach().numpy(),
        connectivity,
        frames=np.arange(k) * config["window"],
        centered=False,
        subtitles=["GMM Cluster: {}".format(i) for i in range(k)],
        title="Mean Sample from GMM Clusters",
        fps=45,
        N_FRAMES=config["window"],
        VID_NAME="cluster.mp4",
        SAVE_ROOT=out_path + "/gen_clips_means/",
    )

if gen_samples_cluster:
    k_pred, gmm = get_gmm_clusters(latents, k, label="cluster", path=out_path)
    assert len(k_pred) == len(dataset)

    gmm_means = torch.tensor(gmm.means_, dtype=torch.float32)
    import pdb

    pdb.set_trace()


#### Generate actions modifying 1 latent dimension at a time
def adjust_single_dim(
    base_latent, latent_means, latent_std, vae, window, mean_offsets, out_path
):
    for i in np.where(latent_std > 0.1)[0]:
        # We take the mean latent of the dataset
        gen_latent = torch.tensor(base_latent, dtype=torch.float32).repeat(3, 1)
        # Add +/- 1.5 to a latent dimension
        gen_latent[[0, 2], i] += torch.tensor([-3, 3])

        synth_rot6d = (
            vae.decoder(gen_latent.cuda()).moveaxis(-1, 1).reshape((3 * window, -1, 6))
        )

        pose = fwd_kin_cont6d_torch(
            synth_rot6d,
            dataset.kinematic_tree,
            mean_offsets,
            torch.zeros((3 * window, 3)),  # root.moveaxis(-1, 1).reshape((-1, 3)),
            do_root_R=True,
        )

        vis.pose.grid3D(
            pose.cpu().detach().numpy(),
            connectivity,
            frames=np.arange(3) * window,
            centered=False,
            labels=["-3", "{:.3f}".format(base_latent[i]), "+3"],
            title="Latent {}: $\mu={:.3f}$, $\sigma={:.3f}$".format(
                i, latent_means[i], latent_std[i]
            ),
            fps=45,
            N_FRAMES=config["window"],
            VID_NAME="latent_{}.mp4".format(i),
            SAVE_ROOT=out_path,
        )


if gen_actions:
    adjust_single_dim(
        latent_means,
        latent_std,
        vae,
        config["window"],
        mean_offsets,
        out_path + "gen_clips_means/",
    )
    adjust_single_dim(
        latents[1000],
        latent_means,
        latent_std,
        vae,
        config["window"],
        mean_offsets,
        out_path + "gen_clips_1K/",
    )
