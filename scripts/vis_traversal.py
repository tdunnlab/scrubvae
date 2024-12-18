import numpy as np
from neuroposelib import visualization as vis
import scrubbed_cvae
from base_path import RESULTS_PATH
import sys
from pathlib import Path
from neuroposelib import read
import torch
from sklearn.linear_model import LinearRegression
from scrubbed_cvae.data.dataset import get_angle2D

### Set/Load Parameters
analysis_key = sys.argv[1]
disentangle_key = "ids"
out_path = RESULTS_PATH + analysis_key
config = read.config(RESULTS_PATH + analysis_key + "/model_config.yaml")
config["model"]["load_model"] = config["out_path"]
vis_decode_path = config["out_path"] + "/vis_decode/"
Path(vis_decode_path).mkdir(parents=True, exist_ok=True)
connectivity = read.connectivity_config(config["data"]["skeleton_path"])
dataset_label = "Train"
### Load Datasets
loader, model = scrubbed_cvae.get.data_and_model(
    config,
    load_model=config["out_path"],
    epoch=sys.argv[2],
    dataset_label=dataset_label,
    data_keys=["x6d", "root", "offsets", disentangle_key],
    # normalize=["avg_speed_3d"],
    shuffle=False,
    verbose=0,
)

latents = scrubbed_cvae.get.latents(
    config=config,
    model=model,
    epoch=sys.argv[2],
    loader=loader,
    device="cuda",
    dataset_label=dataset_label,
)

# dis_w = LinearRegression().fit(latents, loader.dataset[:][disentangle_key]).coef_

n_shifts = 2
# sample_idx = [4000000, 2000000, 3000000, 60000, 1294585]
sample_idx = [1000, 20000, 400000, 600000]
# shift = torch.linspace(0, np.pi, n_shifts)[:, None]
# shift = torch.from_numpy(get_angle2D(shift))
# shift = torch.linspace(-2.5, 5, n_shifts)[:, None]

for sample_i in sample_idx:
    data = loader.dataset[sample_i]
    data = {
        k: v.cuda()[None, ...].repeat(
            (n_shifts + 1,) + tuple(np.ones(len(v.shape), dtype=int))
        )
        for k, v in data.items()
    }
    z_traverse = latents[sample_i : sample_i + 1, :].repeat(n_shifts + 1, 1).cuda()
    # data[disentangle_key][1:, :] += shift.cuda()
    # import pdb; pdb.set_trace()
    data["ids"] = torch.arange(3)[:,None].long()

    data_o = model.decode(z_traverse, data)
    pose = (
        scrubbed_cvae.data.dataset.fwd_kin_cont6d_torch(
            data_o["x6d"].reshape((-1,) + data_o["x6d"].shape[2:]),
            model.kinematic_tree,
            data["offsets"].reshape((-1,) + data["offsets"].shape[2:]),
            root_pos=data_o["root"].reshape((-1, 3)),
            do_root_R=True,
        )
        .detach()
        .cpu()
        .numpy()
    )

    # subtitles = [
    #     "{:2f}".format(val)
    #     for val in data["fluorescence"].detach().cpu().numpy().squeeze()
    # ]

    vis.pose.grid3D(
        pose,
        connectivity,
        frames=np.arange(n_shifts + 1) * model.window,
        centered=False,
        subtitles=None,
        title=dataset_label + " Data - {} Traversal".format(disentangle_key),
        fps=20,
        N_FRAMES=model.window,
        VID_NAME=dataset_label + "grid{}_mod.mp4".format(sample_i),
        SAVE_ROOT=vis_decode_path,
    )

    vis.pose.arena3D(
        pose,
        connectivity,
        frames=np.arange(n_shifts + 1) * model.window,
        centered=False,
        # subtitles=None,
        # title=dataset_label + " Data - {} Traversal".format(disentangle_key),
        fps=20,
        N_FRAMES=model.window,
        VID_NAME=dataset_label + "arena{}_mod.mp4".format(sample_i),
        SAVE_ROOT=vis_decode_path,
    )

    # scrubbed_cvae.eval.traverse_latent(
    #     model,
    #     loader.dataset,
    #     latents,
    #     torch.tensor(dis_w),
    #     sample_i,
    #     connectivity,
    #     label=disentangle_key,
    #     minmax=10,
    #     n_shifts=15,
    #     circle=False,
    #     save_path=vis_decode_path + "{}/".format(disentangle_key),
    # )

    # ## Latent Traversal
    # # norm_z_shift = dis_w / torch.linalg.vector_norm(dis_w,dim=0,keepdim=True)
    # print("Speed Weights Vector Norm: {}".format(torch.linalg.norm(dis_w,dim=-1)))
    # minmax = 10
    # n_shifts = 15
    # graded_z_shift = ( torch.linspace(-minmax, minmax, n_shifts)[:,None].cuda() @ dis_w.sum(dim=0,keepdim=True).cuda() )

    # f = plt.figure(figsize=(15, 15))
    # for sample_i in sample_idx:
    #     sample_latent = latents[sample_i : sample_i + 1].repeat(n_shifts, 1).cuda()
    #     print("Latent Norm: {}".format(torch.linalg.norm(sample_latent[0])))
    #     sample_latent += graded_z_shift

    #     data_o = vae.decode(sample_latent)
    #     offsets = dataset[sample_i]["offsets"].cuda()
    #     pose = (
    #         fwd_kin_cont6d_torch(
    #             data_o["x6d"].reshape((-1, n_keypts, 6)),
    #             vae.kinematic_tree,
    #             offsets.repeat(n_shifts, 1, 1),
    #             root_pos=data_o["root"].reshape((-1, 3)),
    #             do_root_R=True,
    #         )
    #         .cpu()
    #         .detach()
    #         .numpy()
    #     )

    #     # part_lbl = ""
    #     # if config["speed_decoder"] == "part":
    #     #     part_lbl = PARTS[i]

    #     # for vis_plane in ["xz", "xy"]:
    #     #     pose_trans = pose.reshape(n_shifts, config["window"], n_keypts, 3)
    #     #     pose_trans[..., PLANE[vis_plane[-1]]] += +(
    #     #         np.linspace(-20, 20, n_shifts) * n_shifts
    #     #     )[:, None, None]
    #     #     plot.trace(
    #     #         pose_trans.reshape(-1, n_keypts, 3),
    #     #         connectivity,
    #     #         frames=np.arange(n_shifts) * config["window"],
    #     #         n_full_pose=3,
    #     #         vis_plane=vis_plane,
    #     #         centered=False,
    #     #         N_FRAMES=config["window"],
    #     #         FIG_NAME=dataset_label + "{}_trace_{}.png".format(part_lbl, sample_i),
    #     #         SAVE_ROOT=vis_decode_path,
    #     #     )

    #     subtitles = (
    #         (sample_latent @ dis_w.T.cuda()).detach().cpu().numpy()
    #     )
    #     if dis_w.shape[0] != 1:
    #         subtitles = [" ".join(["{:.2f}".format(ss) for ss in s]) for s in subtitles.squeeze()]
    #     else:
    #         subtitles = ["{:.2f}".format(s) for s in subtitles.squeeze()]

    #     vis.pose.grid3D(
    #         pose,
    #         connectivity,
    #         frames=np.arange(n_shifts) * vae.window,
    #         centered=False,
    #         subtitles=subtitles,
    #         title=dataset_label + " Data - {} Traversal".format(k),
    #         fps=15,
    #         N_FRAMES=vae.window,
    #         VID_NAME=dataset_label + "grid{}_mod.mp4".format(sample_i),
    #         SAVE_ROOT=save_path,
    #     )

    #     vis.pose.arena3D(
    #         pose,
    #         connectivity,
    #         frames=np.arange(n_shifts) * vae.window,
    #         centered=False,
    #         fps=15,
    #         N_FRAMES=vae.window,
    #         VID_NAME=dataset_label + "arena{}_mod.mp4".format(sample_i),
    #         SAVE_ROOT=save_path,
    #     )

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
