from data.dataset import MouseDataset, fwd_kin_cont6d_torch, inv_normalize_root
import torch
from dappy import visualization as vis
import numpy as np
from pathlib import Path
import tqdm
import matplotlib.pyplot as plt
import utils
from dappy import read
import torch.optim as optim
from constants import PLANE
import plot

PARTS = ["Root", "Head + Spine", "L-Arm", "R-Arm", "L-Leg", "R-Leg"]

### Set/Load Parameters
path = "gr_parts1/partspd10_rc_w51_b1_midfwd_full"
base_path = "/mnt/home/jwu10/working/behavior_vae/"
config = read.config("/mnt/ceph/users/jwu10/results/vae/" + path + "/model_config.yaml")
config["load_model"] = config["out_path"]
config["load_epoch"] = 130
vis_decode_path = config["out_path"] + "/vis_decode/"
Path(vis_decode_path).mkdir(parents=True, exist_ok=True)
connectivity = read.connectivity_config(base_path + "/configs/mouse_skeleton.yaml")

dataset_label = "Train"
### Load Dataset
dataset = MouseDataset(
    data_path=config["data_path"],
    skeleton_path=base_path + "/configs/mouse_skeleton.yaml",
    train=True,
    window=config["window"],
    stride=1,
    direction_process=config["direction_process"],
    get_speed=config["speed_decoder"],
    get_root=config["arena_size"] is not None,
    get_raw_pose=False,
    arena_size=config["arena_size"],
)
arena_size = None if config["arena_size"] is None else dataset.arena_size.cuda()
kinematic_tree = dataset.kinematic_tree
n_keypts = dataset.n_keypts

vae, speed_decoder, device = utils.init_model(
    config, dataset.n_keypts, config["invariant"]
)
vae.eval()
speed_decoder.eval()
latents = utils.get_latents(vae, dataset, config, device, dataset_label)

if config["gradient_reversal"] == None:
    spd_w = speed_decoder.weight.cpu().detach()
else:
    spd_w = speed_decoder.decoder.weight.cpu().detach()


spd_true = dataset[:]["speed"].mean(axis=1)
if spd_w.shape[0] != 1:
    spd_w_unit = spd_w/torch.linalg.norm(spd_w, axis=-1, keepdim=True)
    spd_w_corr = spd_w_unit @ spd_w_unit.T

    spd_true_unit = spd_true/torch.linalg.norm(spd_true, axis=0, keepdim=True)
    spd_true_corr = spd_true_unit.T @ spd_true_unit
    
    f, ax_arr = plt.subplots(1, 2, figsize=(20, 10))
    im = [ax_arr[0].imshow(spd_w_corr.cpu().detach().numpy())]
    ax_arr[0].set_title("Decoder Weight Correlation")

    im += [ax_arr[1].imshow(spd_true_corr.cpu().detach().numpy())]
    ax_arr[1].set_title("True Speed Correlation")

    for i, ax in enumerate(ax_arr):
        plt.colorbar(im[i], ax=ax)
        if config["speed_decoder"] == "part":
            ax.set_xticks(ticks=range(6), labels=PARTS)
            ax.set_yticks(ticks=range(6), labels=PARTS)

    f.tight_layout()
    plt.savefig(vis_decode_path + "spd_corr.png")
    plt.close()
    
else:
    spd_w = spd_w[None, :]

# if config["speed_decoder"] == "avg":
# avg_spd_w = spd_w.mean(dim=0)
spd_o = latents @ spd_w.T
f, ax = plt.subplots(1, spd_o.shape[-1], figsize=(8 * spd_o.shape[-1], 10))

for i in tqdm.tqdm(range(spd_o.shape[-1])):
    ax[i].scatter(spd_true[:, i], spd_o[:, i])
    if config["speed_decoder"] == "part":
        ax[i].set_title(PARTS[i])
    ax[i].set_xlabel("Speed Target")
    ax[i].set_ylabel("Speed Prediction")

f.tight_layout()
plt.savefig(vis_decode_path + dataset_label + "spd_scatter.png")
plt.close()

print("Min Speed: {}".format(dataset[:]["speed"].min()))

# nrm = (avg_spd_w @ avg_spd_w.t()).ravel()
# z_sub = (latents - (avg_speed_o @ spd_w) / nrm).cuda()

# from sklearn.linear_model import LinearRegression
# from sklearn.metrics import r2_score
# from sklearn.decomposition import PCA
# spd_true = dataset[:]["speed"].mean(dim=-1).detach().cpu().numpy()

# z_sub = PCA(n_components=10).fit_transform(z_sub.detach().cpu().numpy(),)
# zsub_reg = LinearRegression().fit( z_sub, spd_true, )
# spd_weights = zsub_reg.coef_[None,:]
# spd_pred = zsub_reg.predict(z_sub)
# print(r2_score(spd_true, spd_pred))
# # z_sub = z_sub.detach().cpu().numpy()

# for i in range(10):
#     nrm = (spd_weights @ spd_weights.T).ravel()
#     z_sub = (z_sub - (z_sub @ spd_weights.T @ spd_weights) / nrm)

#     zsub_reg = LinearRegression().fit( z_sub, spd_true)
#     spd_weights = zsub_reg.coef_[None,:]
#     spd_pred = zsub_reg.predict(z_sub)
#     print(r2_score(spd_true, spd_pred))

# import pdb; pdb.set_trace()
sample_idx = [2000, 200000, 400000, 600000]
import pdb; pdb.set_trace()
## Decode the different latents
for i in range(spd_w.shape[0]):
    norm_z_shift = spd_w[i] / torch.linalg.vector_norm(spd_w[i])
    print("Speed Weights Vector Norm: {}".format(torch.linalg.norm(spd_w[i])))
    minmax = 10
    n_shifts = 15
    graded_z_shift = (
        torch.linspace(-minmax, minmax, n_shifts).cuda()[:, None]
        * spd_w[i:i+1].cuda()  # norm_z_shift[None, :].cuda()
    )

    f = plt.figure(figsize=(15, 15))
    for sample_i in sample_idx:
        sample_latent = latents[sample_i : sample_i + 1].repeat(n_shifts, 1).cuda()

        print("Latent Norm: {}".format(torch.linalg.norm(sample_latent[0])))
        sample_latent += graded_z_shift

        if vae.in_channels == n_keypts * 6:
            x6d_o = vae.decoder(sample_latent)
            root_o = torch.zeros((n_shifts * config["window"], 3))
        elif vae.in_channels == n_keypts * 6 + 3:
            x_o = vae.decoder(sample_latent)
            x6d_o = x_o[:, :-3, :]
            root_o = x_o[:, -3:, :].moveaxis(-1, 1).reshape(-1, 3)
            root_o = inv_normalize_root(root_o, arena_size)

        offsets = dataset[:]["offsets"][sample_i : sample_i + 1].cuda()

        pose = (
            fwd_kin_cont6d_torch(
                x6d_o.moveaxis(-1, 1).reshape((-1, n_keypts, 6)),
                kinematic_tree,
                offsets.repeat(n_shifts, 1, 1, 1).reshape((-1, n_keypts, 3)),
                root_pos=root_o,
                do_root_R=True,
            )
            .cpu()
            .detach()
            .numpy()
        )

        part_lbl = ""
        if config["speed_decoder"] == "part":
            part_lbl = PARTS[i]

        # for vis_plane in ["xz", "xy"]:
        #     pose_trans = pose.reshape(n_shifts, config["window"], n_keypts, 3)
        #     pose_trans[..., PLANE[vis_plane[-1]]] += +(
        #         np.linspace(-20, 20, n_shifts) * n_shifts
        #     )[:, None, None]
        #     plot.trace(
        #         pose_trans.reshape(-1, n_keypts, 3),
        #         connectivity,
        #         frames=np.arange(n_shifts) * config["window"],
        #         n_full_pose=3,
        #         vis_plane=vis_plane,
        #         centered=False,
        #         N_FRAMES=config["window"],
        #         FIG_NAME=dataset_label + "{}_trace_{}.png".format(part_lbl, sample_i),
        #         SAVE_ROOT=vis_decode_path,
        #     )

        subtitles = (sample_latent @ spd_w[i:i+1].T.cuda()).detach().cpu().numpy()
        subtitles = ["{:.2f}".format(s) for s in subtitles.squeeze()]

        vis.pose.grid3D(
            pose,
            connectivity,
            frames=np.arange(n_shifts) * config["window"],
            centered=False,
            subtitles=subtitles,
            title=dataset_label + " Data - Speed Latent Shift",
            fps=15,
            N_FRAMES=config["window"],
            VID_NAME=dataset_label + "grid{}_mod{}.mp4".format(part_lbl, sample_i),
            SAVE_ROOT=vis_decode_path,
        )

        vis.pose.arena3D(
            pose,
            connectivity,
            frames=np.arange(n_shifts) * config["window"],
            centered=False,
            # subtitles=np.linspace(-minmax, minmax, n_shifts).astype(str),
            # title=dataset_label + " Data - Speed Latent Shift",
            fps=15,
            N_FRAMES=config["window"],
            VID_NAME=dataset_label + "arena{}_mod{}.mp4".format(part_lbl, sample_i),
            SAVE_ROOT=vis_decode_path,
        )

    #     graded_speeds = (
    #         np.diff(pose.reshape(n_shifts, config["window"], -1, 3), axis=1) ** 2
    #     )
    #     graded_speeds = np.sqrt(graded_speeds.sum(axis=-1)).mean(axis=(-1, -2))

    #     plt.plot(
    #         np.linspace(-minmax, minmax, n_shifts), graded_speeds, label=str(sample_i)
    #     )

    # plt.xlabel("Z Shift Scale")
    # plt.ylabel("Speed of Reconstruction")
    # plt.legend()
    # plt.savefig(vis_decode_path + dataset_label + "spd_graded_large.png")
    # plt.close()
