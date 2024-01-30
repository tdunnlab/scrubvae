import ssumo
from torch.utils.data import DataLoader
from dappy import read
import torch
from dappy import visualization as vis
import numpy as np
from pathlib import Path

from base_path import RESULTS_PATH

z_null = "avg_speed"
gen_means_cluster = False
gen_samples_cluster = False
gen_actions = False
vis_clusters = True

path = "/heading/balanced/"
vis_path = RESULTS_PATH + path + "/vis_latents_300/"
config = read.config(RESULTS_PATH + path + "/model_config.yaml")
k = 25  # Number of clusters
config["model"]["load_model"] = config["out_path"]
config["model"]["start_epoch"] = 300
config["data"]["stride"] = 1
# config["speed_decoder"] = None
Path(vis_path).mkdir(parents=True, exist_ok=True)

connectivity = read.connectivity_config(config["data"]["skeleton_path"])
dataset_label = "Train"
### Load Datasets
dataset = ssumo.data.get_mouse(
    data_config=config["data"],
    window=config["model"]["window"],
    train=dataset_label == "Train",
    data_keys=["x6d", "root", "offsets", "raw_pose"]
    + config["disentangle"]["features"],
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

latents = ssumo.eval.get.latents(vae, dataset, config, device, dataset_label)
mean_offsets = dataset.data["offsets"].mean(axis=(0, -1)).cuda()
latent_means = latents.mean(axis=0)
latent_std = latents.std(axis=0)
num_latents = latents.shape[-1]

if z_null is not None:
    latents = ssumo.eval.project_to_null(
        latents, vae.disentangle[z_null].decoder.weight.cpu().detach().numpy()
    )[0]

### Visualize clusters
if vis_clusters:
    label = "z{}".format("" if z_null is None else "_" + z_null)
    k_pred, gmm = ssumo.eval.cluster.gmm(
        latents=latents,
        n_components=k,
        label=label,
        path=vis_path,
        covariance_type="full",
    )

    print(
        np.histogram(
            k_pred, bins=len(np.unique(k_pred)), range=(-1.5, np.max(k_pred) + 0.5)
        )[0]
    )

    ssumo.plot.sample_clusters(
        dataset[:]["raw_pose"].detach().cpu().numpy(),
        k_pred,
        connectivity,
        "{}/vis_clusters_{}/".format(vis_path, "" if z_null is None else z_null),
    )

import pdb

pdb.set_trace()

if gen_means_cluster:
    k_pred, gmm = utils.get_gmm_clusters(latents, k, label="cluster", path=vis_path)
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
        SAVE_ROOT=vis_path + "/gen_clips_means/",
    )

if gen_samples_cluster:
    k_pred, gmm = get_gmm_clusters(latents, k, label="cluster", path=vis_path)
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
        vis_path + "gen_clips_means/",
    )
    adjust_single_dim(
        latents[1000],
        latent_means,
        latent_std,
        vae,
        config["window"],
        mean_offsets,
        vis_path + "gen_clips_1K/",
    )
