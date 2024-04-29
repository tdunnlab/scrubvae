import numpy as np
import matplotlib.pyplot as plt
import ssumo
from base_path import RESULTS_PATH, CODE_PATH
import sys
from pathlib import Path
from dappy import read

palette = ssumo.plot.constants.PALETTE_2
experiment_folder = sys.argv[1]
method = sys.argv[2]
task_id = sys.argv[3] if len(sys.argv) > 3 else ""

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

dataset_label = "Train"
if method == "log_class":
    disentangle_keys = ["ids"]
else:
    disentangle_keys = ["avg_speed_3d", "heading"]

metrics = {}
for an_key in analysis_keys:
    folder = "{}/{}/".format(RESULTS_PATH, an_key)
    print("Reading in folder: {}".format(folder))
    metrics[an_key] = ssumo.eval.metrics.epoch_regression(
        folder, method, dataset_label, save_load=True, disentangle_keys=disentangle_keys
    )

rows = 1 if method=="log_class" else 2
if task_id == "":
    ## Plot R^2
    for key in disentangle_keys:
        f, ax_arr = plt.subplots(rows, 1, figsize=(15, 15))
        plt.title("R2 of Regression Using {}".format(method.title()))
        for path_i, p in enumerate(analysis_keys):
            for i, metric in enumerate(metrics[p][key].keys()):
                if rows == 1:
                    ax = ax_arr
                else:
                    ax = ax_arr[i]

                ax.plot(
                    metrics[p]["epochs"],
                    metrics[p][key][metric],
                    label="{}".format(p),
                    color=palette[path_i],
                    alpha=0.5,
                )

                ax.set_ylabel(metric)
                ax.legend()
                ax.set_xlabel("Epoch")
                ax.set_ylim(bottom=max(min(metrics[p][key][metric]), 0))

                ax.set_ylim(bottom=0, top=1)

        f.tight_layout()
        plt.savefig("{}/{}_{}_epoch.png".format(out_path, key, method))
        plt.close()
