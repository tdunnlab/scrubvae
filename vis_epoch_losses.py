from base_path import RESULTS_PATH, CODE_PATH
import matplotlib.pyplot as plt
from dappy import read
from pathlib import Path
import numpy as np
import pickle
import sys


def plot_loss_curve(
    losses, dataset_label="Train", loss_key="jpe", ylimit=False, path="./"
):
    plt.figure(figsize=(10, 5))
    plt.title("{} {} Loss".format(dataset_label, loss_key.title()))
    for k, v in losses.items():

        plt.plot(
            np.arange(len(v)),
            v,
            label=k,
            alpha=0.5,
            linewidth=1,
        )
    plt.yscale("log")
    if ylimit:
        plt.ylim(ylimit[0], ylimit[1])
    plt.xlabel("Epoch")
    plt.ylabel("Log Loss")
    plt.legend()
    plt.savefig(path)
    plt.close()


experiment_folder = sys.argv[1]
loss_key = sys.argv[2]
task_id = sys.argv[3] if len(sys.argv) > 3 else ""
if experiment_folder.endswith(".yaml"):
    config_path = CODE_PATH + "/configs/" + experiment_folder
    config = read.config(config_path)
    analysis_keys = config["PATHS"]
    out_path = config["OUT"]
elif Path(RESULTS_PATH + experiment_folder).is_dir():
    # If experiment folder is a single directory
    analysis_keys = [experiment_folder]
    out_path = RESULTS_PATH + experiment_folder

if task_id.isdigit():
    analysis_keys = [analysis_keys[int(task_id)]]

losses = {}
for an_key in analysis_keys:
    try:
        loss_dict = pickle.load(
            open("{}/{}/losses/loss_dict.p".format(RESULTS_PATH, an_key), "rb")
        )
    except:
        loss_dict = pickle.load(
            open("{}/{}/losses/loss_dict_Train.p".format(RESULTS_PATH, an_key), "rb")
        )

    losses[an_key] = loss_dict[loss_key]

plot_loss_curve(
    losses,
    dataset_label="Train",
    loss_key=loss_key,
    ylimit=False,
    path="{}/{}_epoch_loss.png".format(out_path, loss_key),
)
