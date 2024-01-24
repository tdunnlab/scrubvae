from ssumo.data.dataset import fwd_kin_cont6d_torch
from torch.utils.data import DataLoader
import ssumo
import torch
from dappy import visualization as vis
import numpy as np
from pathlib import Path
import tqdm
import matplotlib.pyplot as plt
from dappy import read

PARTS = ["Root", "Head + Spine", "L-Arm", "R-Arm", "L-Leg", "R-Leg"]

### Set/Load Parameters
path = "gre1_b1_true_x360"
base_path = "/mnt/ceph/users/jwu10/results/vae/heading/"
config = read.config(base_path + path + "/model_config.yaml")
config["model"]["load_model"] = config["out_path"]
config["model"]["start_epoch"] = 180
vis_decode_path = config["out_path"] + "/vis_decode/"
Path(vis_decode_path).mkdir(parents=True, exist_ok=True)
connectivity = read.connectivity_config(config["data"]["skeleton_path"])

dataset_label = "Train"
### Load Dataset
dataset = ssumo.data.get_mouse(
    data_config=config["data"],
    window=config["model"]["window"],
    train=dataset_label == "Train",
    data_keys=["x6d", "root", "offsets"]
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

latents = ssumo.evaluate.get.latents(vae, dataset, config, device, dataset_label)

for k in ["avg_speed","heading"]:#vae.disentangle_keys:
    save_path = vis_decode_path + "{}/".format(k)
    Path(save_path).mkdir(parents=True, exist_ok=True)
    dis_w = vae.disentangle[k].decoder.weight.cpu().detach()
    pred_true = dataset[:][k]

    if dis_w.shape[0] != 1:
        dis_w_unit = dis_w / torch.linalg.norm(dis_w, axis=-1, keepdim=True)
        dis_w_corr = dis_w_unit @ dis_w_unit.T

        pred_true_unit = pred_true / torch.linalg.norm(pred_true, axis=0, keepdim=True)
        pred_true_corr = pred_true_unit.T @ pred_true_unit

        f, ax_arr = plt.subplots(1, 2, figsize=(20, 10))
        im = [ax_arr[0].imshow(dis_w_corr.cpu().detach().numpy())]
        ax_arr[0].set_title("Decoder Weight Correlation")

        im += [ax_arr[1].imshow(pred_true_corr.cpu().detach().numpy())]
        ax_arr[1].set_title("True Label Correlation")

        for i, ax in enumerate(ax_arr):
            plt.colorbar(im[i], ax=ax)
            # if config["speed_decoder"] == "part":
            #     ax.set_xticks(ticks=range(6), labels=PARTS)
            #     ax.set_yticks(ticks=range(6), labels=PARTS)

        f.tight_layout()
        plt.savefig(save_path + "spd_corr.png")
        plt.close()



    pred = latents @ dis_w.T
    f, ax = plt.subplots(1, pred.shape[-1], figsize=(8 * pred.shape[-1], 10))
    for i in tqdm.tqdm(range(pred.shape[-1])):
        if pred.shape[-1] > 1:
            ax_obj = ax[i]
        else:
            ax_obj = ax
        ax_obj.scatter(pred_true[:, i], pred[:, i])
        # if config["speed_decoder"] == "part":
        ax_obj.set_title("Latent {} Dim {}".format(k, i))
        ax_obj.set_xlabel("Target")
        ax_obj.set_ylabel("Prediction")

    f.tight_layout()
    plt.savefig(save_path + "{}{}_scatter.png".format(dataset_label, k))
    plt.close()

    print("Min {}: {}".format(k,pred_true.min()))
    print("Max {}: {}".format(k,pred_true.max()))

    sample_idx = [2000, 200000, 400000, 600000]

    ## Latent Traversal
    # norm_z_shift = dis_w / torch.linalg.vector_norm(dis_w,dim=0,keepdim=True)
    print("Speed Weights Vector Norm: {}".format(torch.linalg.norm(dis_w,dim=-1)))
    minmax = 10
    n_shifts = 15
    graded_z_shift = ( torch.linspace(-minmax, minmax, n_shifts)[:,None].cuda() @ dis_w.sum(dim=0,keepdim=True).cuda() )

    f = plt.figure(figsize=(15, 15))
    for sample_i in sample_idx:
        sample_latent = latents[sample_i : sample_i + 1].repeat(n_shifts, 1).cuda()
        print("Latent Norm: {}".format(torch.linalg.norm(sample_latent[0])))
        sample_latent += graded_z_shift

        data_o = vae.decode(sample_latent)
        offsets = dataset[sample_i]["offsets"].cuda()
        pose = (
            fwd_kin_cont6d_torch(
                data_o["x6d"].reshape((-1, n_keypts, 6)),
                vae.kinematic_tree,
                offsets.repeat(n_shifts, 1, 1),
                root_pos=data_o["root"].reshape((-1, 3)),
                do_root_R=True,
            )
            .cpu()
            .detach()
            .numpy()
        )

        # part_lbl = ""
        # if config["speed_decoder"] == "part":
        #     part_lbl = PARTS[i]

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

        subtitles = (
            (sample_latent @ dis_w.T.cuda()).detach().cpu().numpy()
        )
        if dis_w.shape[0] != 1:
            subtitles = [" ".join(["{:.2f}".format(ss) for ss in s]) for s in subtitles.squeeze()]
        else:
            subtitles = ["{:.2f}".format(s) for s in subtitles.squeeze()]

        vis.pose.grid3D(
            pose,
            connectivity,
            frames=np.arange(n_shifts) * vae.window,
            centered=False,
            subtitles=subtitles,
            title=dataset_label + " Data - {} Traversal".format(k),
            fps=15,
            N_FRAMES=vae.window,
            VID_NAME=dataset_label + "grid{}_mod.mp4".format(sample_i),
            SAVE_ROOT=save_path,
        )

        vis.pose.arena3D(
            pose,
            connectivity,
            frames=np.arange(n_shifts) * vae.window,
            centered=False,
            fps=15,
            N_FRAMES=vae.window,
            VID_NAME=dataset_label + "arena{}_mod.mp4".format(sample_i),
            SAVE_ROOT=save_path,
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

# nrm = (avg_dis_w @ avg_dis_w.t()).ravel()
# z_sub = (latents - (avg_speed_o @ dis_w) / nrm).cuda()

# from sklearn.linear_model import LinearRegression
# from sklearn.metrics import r2_score
# from sklearn.decomposition import PCA
# pred_true = dataset[:]["speed"].mean(dim=-1).detach().cpu().numpy()

# z_sub = PCA(n_components=10).fit_transform(z_sub.detach().cpu().numpy(),)
# zsub_reg = LinearRegression().fit( z_sub, pred_true, )
# dis_weights = zsub_reg.coef_[None,:]
# spd_pred = zsub_reg.predict(z_sub)
# print(r2_score(pred_true, spd_pred))
# # z_sub = z_sub.detach().cpu().numpy()

# for i in range(10):
#     nrm = (dis_weights @ dis_weights.T).ravel()
#     z_sub = (z_sub - (z_sub @ dis_weights.T @ dis_weights) / nrm)

#     zsub_reg = LinearRegression().fit( z_sub, pred_true)
#     dis_weights = zsub_reg.coef_[None,:]
#     spd_pred = zsub_reg.predict(z_sub)
#     print(r2_score(pred_true, spd_pred))
