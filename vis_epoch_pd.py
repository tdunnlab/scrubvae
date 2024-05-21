import numpy as np
import matplotlib.pyplot as plt
import ssumo
from base_path import RESULTS_PATH, CODE_PATH
import sys
from pathlib import Path
from dappy import read
import pickle
from ssumo.eval.metrics import custom_cv_5folds
from sklearn.linear_model import LogisticRegression

palette = ssumo.plot.constants.PALETTE_2
experiment_folder = sys.argv[1]
task_id = sys.argv[2] if len(sys.argv) > 2 else ""

if experiment_folder.endswith(".yaml"):
    config_path = CODE_PATH + "/configs/" + experiment_folder
    config = read.config(config_path)
    analysis_keys = config["PATHS"]
    out_path = config["OUT"]
elif Path(RESULTS_PATH + experiment_folder).is_dir():
    analysis_keys = [experiment_folder]
    out_path = RESULTS_PATH + experiment_folder

if task_id.isdigit():
    analysis_keys = [analysis_keys[int(task_id)]]

config = read.config(RESULTS_PATH + analysis_keys[0] + "/model_config.yaml")
loader = ssumo.get.mouse_data(
    data_config=config["data"],
    window=config["model"]["window"],
    train=True,
    data_keys=[
        "x6d",
        "root",
        "avg_speed_3d",
        "fluorescence",
    ],
    shuffle=False,
    normalize=["avg_speed_3d"],
)
pd_label = np.array(loader.dataset[:]["fluorescence"] < 0.9, dtype=int).ravel()
speed = loader.dataset[:]["avg_speed_3d"].cpu().detach().numpy()

metrics_full = {}
for an_key in analysis_keys:
    path = "{}/{}/".format(RESULTS_PATH, an_key)
    metrics = pickle.load(open(path + "pd_Train.p", "rb"))
    config = read.config(path + "/model_config.yaml")
    config["model"]["load_model"] = config["out_path"]

    pickle_path = "{}/pd_{}.p".format(config["out_path"], "Train")
    if Path(pickle_path).is_file():
        metrics = pickle.load(open(pickle_path, "rb"))
        epochs_to_test = [
            e for e in ssumo.get.all_saved_epochs(path) if e not in metrics["epochs"]
        ]
        metrics["epochs"] = np.concatenate(
            [metrics["epochs"], epochs_to_test]
        ).astype(int)
    else:
        metrics = {k: [] for k in ["Latents", "Both"]}
        metrics["epochs"] = ssumo.get.all_saved_epochs(path)
        epochs_to_test = metrics["epochs"]

    for _, epoch in enumerate(epochs_to_test):

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

        z = ssumo.get.latents(config, model, epoch, loader, "cuda", "Train")
        clf = LogisticRegression(solver="sag", max_iter=200)
        metrics["Latents"] += [cross_val_score(clf, z, pd_label).mean()]
        # metrics["Both"] += [cross_val_score(clf, np.concatenate([z, speed], axis=-1), pd_label).mean()]

    pickle.dump(
        metrics,
        open(pickle_path, "wb"),
    )
    metrics_full[an_key] = metrics


# speed_preds = cross_val_score(clf, speed, pd_label).mean()
if task_id == "":
    ## Plot R^2
    f = plt.figure(figsize=(15, 10))
    plt.title("5-Fold Logistic PD Classification")
    max_epochs = 0
    for path_i, p in enumerate(analysis_keys):
        for i, metric in enumerate(metrics_full[p].keys()):
            if metric == "epochs":
                continue

            plt.plot(
                metrics_full[p]["epochs"],
                metrics_full[p][metric],
                label="{} {}".format(p, metric),
                color=palette[path_i*2 + i],
                alpha=0.5,
            )
            max_epochs = max(max_epochs, metrics_full[p]["epochs"].max())

    # plt.plot(
    #     np.arange(max_epochs),
    #     np.ones(max_epochs)*speed_preds,
    #     label="Speed Only",
    #     color=palette[-1],
    #     alpha=0.5,
    # )

    plt.ylabel("Accuracy")
    plt.legend()
    plt.xlabel("Epoch")
    # plt.ylim(bottom=max(min(metrics_full[p][metric]), 0))
    plt.ylim(bottom=0.5, top=1)

    f.tight_layout()
    plt.savefig("{}/speed_pd_epoch.png".format(out_path))
    plt.close()
