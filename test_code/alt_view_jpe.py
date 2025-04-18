import ssumo
from neuroposelib import read
import numpy as np
import matplotlib.pyplot as plt
import sys
import torch
from ssumo.get.data import get_projected_2D_kinematics
from ssumo.train.trainer import predict_batch
from ssumo.train.losses import mpjpe_loss
from tqdm import tqdm
import gc

# from base_path import RESULTS_PATH

RESULTS_PATH = "/mnt/ceph/users/hkoneru/results/vae/"
maxlabels = 19
stride = 10
angles = 19
redo_latent = False

analysis_key = sys.argv[1]
config = read.config(RESULTS_PATH + analysis_key + "/model_config.yaml")
# config["data"]["batch_size"] = 1024
all_angles = [
    np.round(np.array([0, -np.sin(i), -np.cos(i)]), 10)
    for i in np.linspace(0, np.pi, angles)
]
jpe_grid = [[0 for j in range(angles)] for i in range(angles)]

with torch.no_grad():
    for a in range(angles):
        config["data"]["project_axis"] = [all_angles[a]]
        config["data"]["stride"] = stride

        dataset_label = "Train"
        ### Load Datasets
        loader, model = ssumo.get.data_and_model(
            config,
            load_model=config["out_path"],
            epoch=sys.argv[2],
            dataset_label=dataset_label,
            data_keys=["x6d", "view_axis", "offsets", "target_pose"],
            shuffle=False,
            verbose=0,
        )
        # data = {k: v.to("cuda") for k, v in data.items()}
        # data = get_projected_2D_kinematics(
        #     data,
        #     all_angles[a][None, :].repeat(len(data["raw_pose"]), 0),
        #     skeleton_config,
        # )
        skeleton_config = read.config(config["data"]["skeleton_path"])
        for alt in tqdm(range(angles)):
            jpe_list = []
            for batch_idx, data in enumerate(loader):
                axis = (
                    torch.tensor(
                        all_angles[alt][None, :].repeat(len(data["raw_pose"]), 0)
                    )
                    .to("cuda")
                    .to(torch.float)
                )
                data["view_axis"] = axis
                data = {k: v.to("cuda") for k, v in data.items()}
                data["target_pose"] = get_projected_2D_kinematics(
                    {k: data[k] for k in ["raw_pose", "target_pose"]},
                    axis,
                    skeleton_config,
                )["target_pose"]
                data_o = predict_batch(
                    model,
                    data,
                    model.disentangle_keys
                    + [
                        "segment_lens"
                        for i in [1]
                        if config["data"].get("segment_lens")
                    ],
                )
                offsets = data["offsets"]
                if "segment_lens" in data_o.keys():
                    offsets = offsets * data_o["segment_lens"][..., None].repeat(
                        1, 1, 1, 3
                    )
                jpe_list.append(
                    mpjpe_loss(
                        data[
                            "target_pose"
                        ],  # data["x6d"].reshape(-1, *data["x6d"].shape[-2:]),
                        data_o["x6d"],
                        model.kinematic_tree,
                        offsets,
                    )
                )
                del data_o
                torch.cuda.synchronize()  # Ensure async ops complete
                gc.collect()  # Python garbage collection
                torch.cuda.empty_cache()  # Release unused memory
            jpe_grid[a][alt] = (sum(jpe_list) / len(jpe_list)).item()
        torch.cuda.synchronize()  # Ensure async ops complete
        gc.collect()
        torch.cuda.empty_cache()

print(jpe_grid)

feat = [
    np.round(np.abs(np.arctan2(all_angles[i][1], all_angles[i][2])), 2)
    for i in range(len(all_angles))
]

stepsize = int(np.ceil(angles / maxlabels))

plt.imshow(jpe_grid)
ax = plt.gca()
# plt.xticks(np.arange(len(axes)))
# plt.yticks(np.arange(len(axes)))
# ax.set_xticklabels(feat)
# ax.set_yticklabels(feat)
plt.xticks(np.arange(angles))
plt.xticks(rotation=90)
plt.yticks(np.arange(angles))
ax.set_xticklabels([f if f in feat[::stepsize] + [feat[-1]] else "" for f in feat])
ax.set_yticklabels([f if f in feat[::stepsize] + [feat[-1]] else "" for f in feat])
plt.colorbar()
plt.savefig("{}alt_view_jpe.png".format(config["out_path"]), dpi=400)
plt.close()
