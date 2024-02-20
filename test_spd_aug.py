import numpy as np
from pathlib import Path
from dappy import read
import tqdm
from sklearn.decomposition import PCA
from data.dataset import MouseDataset
import utils
from torch.utils.data import DataLoader
from dappy import read, write
from dappy import visualization as vis
import scipy.linalg as spl

path = "avgspd_ndgre1_rc_w51_b1_midfwd_full_a05"
base_path = "/mnt/ceph/users/jwu10/results/vae/gr_scratch/"
config = read.config(base_path + path + "/model_config.yaml")
config["load_model"] = config["out_path"]
config["load_epoch"] = 470
aug_path = config["out_path"] + "/spd_aug/"
Path(aug_path).mkdir(parents=True, exist_ok=True)
connectivity = read.connectivity_config(
    "/mnt/home/jwu10/working/behavior_vae/configs/mouse_skeleton.yaml"
)

# from dappy import augmentations
# pose, ids = read.pose_h5(config["data_path"])
# meta, _ = read.meta("/mnt/home/jwu10/working/ceph/data/ensemble_healthy/metadata.csv", id=ids)
aug_levels = np.linspace(0.5, 1.5, 7)
# pose_aug, meta_aug, ids_aug = augmentations.speed(pose, aug_levels, ids, meta)

# write.pose_h5(pose_aug, ids_aug, aug_path + "/pose_aug.h5")
# meta_aug.to_csv(aug_path + "/metadata.csv")

train_ids = np.arange(21).reshape(3, 7)
vae, spd_decoder, device = utils.init_model(config, 18, config["conditional"])
vae.eval()
# raw_pose = []
# for i, id_set in enumerate(train_ids):
#     dataset = MouseDataset(
#         data_path="/mnt/home/jwu10/working/ceph/results/vae/avgspd_rc_w51_midfwd_full/spd_aug/pose_aug.h5",
#         skeleton_path="/mnt/home/jwu10/working/behavior_vae/configs/mouse_skeleton.yaml",
#         train=True,
#         train_ids=id_set,
#         window=config["window"],
#         stride=config["stride"],
#         direction_process=config["direction_process"],
#         get_speed=False,
#         get_raw_pose=True,
#         get_root=True,
#         arena_size=config["arena_size"],
#         conditional=None,
#         remove_speed_outliers=None,
#     )
#     # raw_pose += [dataset[:]["raw_pose"]]

#     latents = utils.get_latents(
#         vae, dataset, config, device, "spd_aug_train_a{}".format(i)
#     )

# raw_pose = np.concatenate(raw_pose,axis=0)
latents = []
for i in range(3):
    latents += [
        np.load(
            "{}/latents/spd_aug_train_a{}_{}.npy".format(
                config["out_path"], i, config["load_epoch"]
            )
        )
    ]

latents = np.concatenate(latents)
k_preds, model = utils.get_gmm_clusters(
    latents[::5],
    50,
    label="z_gmm_full",
    path=aug_path,
    covariance_type="full",
)
k_preds = model.predict(latents)

if "gr" in path:
    spd_weights = spd_decoder.decoder.weight.cpu().detach().numpy()
else:
    spd_weights = spd_decoder.weight.cpu().detach().numpy()

U_orth = spl.null_space(spd_weights)
latents = latents @ U_orth
# nrm = (spd_weights @ spd_weights.T).ravel()
# avg_spd_o = latents @ spd_weights.T
# latents = latents - (avg_spd_o @ spd_weights) / nrm

k_preds_sub, model = utils.get_gmm_clusters(
    latents[::5],
    50,
    label="zsub_gmm_full",
    path=aug_path,
    covariance_type="full",
)
k_preds_sub = model.predict(latents)


# for j, preds in enumerate([k_preds, k_preds_sub]):
#     ### Sample 9 videos from each cluster
#     n_samples = 9
#     indices = np.arange(len(preds))
#     for cluster in range(50):
#         label_idx = indices[preds == cluster]
#         num_points = min(len(label_idx), n_samples)
#         permuted_points = np.random.permutation(label_idx)
#         sampled_points = []
#         for i in range(len(permuted_points)):
#             if len(sampled_points) == num_points:
#                 break
#             elif any(np.abs(permuted_points[i] - np.array(sampled_points)) < 100):
#                 continue
#             else:
#                 sampled_points += [permuted_points[i]]

#         print("Plotting Poses from Cluster {}".format(cluster))
#         print(sampled_points)

#         num_points = len(sampled_points)

#         sampled_pose = raw_pose[sampled_points].reshape(
#             num_points * config["window"], dataset.n_keypts, 3
#         )

#         vis.pose.arena3D(
#             sampled_pose,
#             connectivity,
#             frames=np.arange(num_points) * config["window"],
#             centered=False,
#             N_FRAMES=config["window"],
#             fps=30,
#             dpi=100,
#             VID_NAME="cluster{}.mp4".format(cluster),
#             SAVE_ROOT=aug_path + "/sampled_clusters{}/".format(j),
#         )

pose, ids = read.pose_h5(
    "/mnt/home/jwu10/working/ceph/results/vae/avgspd_rc_w51_midfwd_full/spd_aug/pose_aug.h5"
)
pose = pose[np.in1d(ids, train_ids), ...]
ids = ids[np.in1d(ids, train_ids)]
window_ids = ids.copy()
window_pose = pose.copy()
for i in np.unique(window_ids):
    # del_ids = np.where(window_ids == i)[0][: 2 * (config["window"] // 2)]

    del_inds = np.concatenate(
        [
            np.where(window_ids == i)[0][: (config["window"] // 2)],
            np.where(window_ids == i)[0][-(config["window"] // 2) :],
        ]
    )
    window_ids = np.delete(window_ids, del_inds)
    window_pose = np.delete(window_pose, del_inds, axis=0)


window_pose, window_ids = window_pose.reshape((3, -1, 18, 3)), window_ids.reshape(
    (3, -1)
)

for preds in [k_preds, k_preds_sub]:
    x1 = preds.reshape((3, -1))[:, np.where(window_ids == 3)[1]]
    accuracy = []
    for i in range(len(aug_levels)):
        xaug = preds.reshape((3, -1))[:, np.where(window_ids == i)[1]]

        if aug_levels[i] <= 1:
            window_cutoff = int(np.round((config["window"] // 2) / aug_levels[i])) - (
                config["window"] // 2
            )
            ind = np.round( np.linspace( window_cutoff, xaug.shape[1] - window_cutoff - 1, x1.shape[1], ) ).astype(int)
            accuracy += [(xaug[:, ind] == x1).sum()]
            accuracy[-1] /= xaug[:, ind].size

            pose_aug = window_pose[:, np.where(window_ids == i)[1], ...][:, ind, ...]
            pose_orig = window_pose[:, np.where(window_ids == 3)[1], ...]
            print(np.linalg.norm(pose_aug - pose_orig, axis=(2,3)).mean())

        elif aug_levels[i] > 1:
            window_cutoff = int(np.round((config["window"] // 2) * aug_levels[i])) - (
                config["window"] // 2
            )
            ind = np.round( np.linspace( window_cutoff, x1.shape[1] - window_cutoff - 1, xaug.shape[1], ) ).astype(int)
            accuracy += [(x1[:, ind] == xaug).sum()]
            accuracy[-1] /= x1[:, ind].size

            pose_aug = window_pose[:, np.where(window_ids == i)[1], ...]
            pose_orig = window_pose[:, np.where(window_ids == 3)[1], ...][:, ind, ...]
            print(np.linalg.norm(pose_aug - pose_orig, axis=(2,3)).mean())
    
    print(accuracy)

