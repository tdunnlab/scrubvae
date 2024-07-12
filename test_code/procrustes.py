import numpy as np
import matplotlib.pyplot as plt
import ssumo
# from base_path import RESULTS_PATH, CODE_PATH
import sys
from pathlib import Path
from neuroposelib import read
from scipy.spatial import procrustes

RESULTS_PATH = "/mnt/ceph/users/jwu10/results/vae/"
palette = ssumo.plot.constants.PALETTE_2
task_id = sys.argv[1] if len(sys.argv)>1 else ""
analysis_keys = ["vanilla_64", "cvae_64", "mi_64_new", "mals_64"]#, "mi_64_new", "mi_64_fixed"]

if task_id.isdigit():
    analysis_keys = [analysis_keys[int(task_id)]]

config = read.config(RESULTS_PATH + analysis_keys[0] + "/1/model_config.yaml")
loader = ssumo.get.mouse_data(
    data_config=config["data"],
    window=config["model"]["window"],
    train=True,
    data_keys=[
        "x6d",
        "root",
        "avg_speed_3d",
    ],
    shuffle=False,
)
epochs = range(60, 80, 5)
distance = {k:[] for k in analysis_keys}
for epoch in epochs:
    for an_key in analysis_keys:
        z = []
        metric = []
        for model_i in range(1,6):
            path = "{}/{}/{}/".format(RESULTS_PATH, an_key, model_i)
            # metrics = pickle.load(open(path + "procrustes_Train.p", "rb"))
            config = read.config(path + "/model_config.yaml")
            config["model"]["load_model"] = config["out_path"]

            pickle_path = "{}/procrustes_{}.p".format(config["out_path"], "Train")

            model = ssumo.get.model(
                model_config=config["model"],
                load_model=config["out_path"],
                epoch=epoch,
                disentangle_config=config["disentangle"],
                n_keypts=loader.dataset.n_keypts,
                direction_process=config["data"]["direction_process"],
                loss_config = config["loss"],
                arena_size=loader.dataset.arena_size,
                kinematic_tree=loader.dataset.kinematic_tree,
                bound=config["data"]["normalize"] is not None,
                discrete_classes=loader.dataset.discrete_classes,
                verbose=-1,
            )
            z_temp = ssumo.get.latents(config, model, epoch, loader, "cuda", "Train")

            if "vanilla" not in an_key:
                z_temp = np.concatenate([z_temp, loader.dataset[:]["avg_speed_3d"]], axis=-1)

            z += [z_temp]

        for first_ind in range(5):
            for second_ind in range(first_ind+1,5):
                metric += [procrustes(z[first_ind], z[second_ind])[2]]

        distance[an_key] += [np.mean(metric)]
    print(distance)
        # pickle.dump(
        #     metrics,
        #     open(pickle_path, "wb"),
        # )

if task_id == "":
    ## Plot R^2
    f = plt.figure(figsize=(15, 10))
    plt.title("Mean Paired Procrustes Between 5 Models")
    max_epochs = 0
    for path_i, p in enumerate(analysis_keys):

        plt.plot(
            epochs,
            distance[p],
            label=p,
            color=palette[path_i],
            alpha=0.5,
        )
        # max_epochs = max(max_epochs, metrics_full[p]["epochs"].max())

    # plt.plot(
    #     np.arange(max_epochs),
    #     np.ones(max_epochs)*speed_preds,
    #     label="Speed Only",
    #     color=palette[-1],
    #     alpha=0.5,
    # )

    plt.ylabel("Procrustes Distance")
    plt.legend()
    plt.xlabel("Epoch")
    # plt.ylim(bottom=max(min(metrics_full[p][metric]), 0))
    # plt.ylim(bottom=0.5, top=1)

    f.tight_layout()
    plt.savefig("/mnt/home/jwu10/working/ssumo/test_code/procrustes.png")
    plt.close()
