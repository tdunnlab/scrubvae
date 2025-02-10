import numpy as np
import matplotlib.pyplot as plt
import scrubvae
from base_path import RESULTS_PATH, CODE_PATH
import sys
from pathlib import Path
from neuroposelib import read

regression_methods = [
    "log_class",
    "linear",
    "mlp",
    "log_class_rand_cv",
    "qda_rand_cv",
    "linear_rand_cv",
    "mlp_rand_cv",
    "log_class_cv",
    "linear_cv",
    "mlp_cv",
]

palette = scrubvae.plot.constants.PALETTE_2
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

if method in regression_methods:
    if ("log_class" in method) or ("qda" in method):
        disentangle_keys = ["ids"]
    elif "1d" in experiment_folder:
        disentangle_keys = ["avg_speed", "heading"]
    else:
        disentangle_keys = ["avg_speed_3d", "heading"]

    print(disentangle_keys)
    metrics = {}
    for an_key in analysis_keys:
        folder = "{}/{}/".format(RESULTS_PATH, an_key)
        print("Reading in folder: {}".format(folder))
        metrics[an_key] = scrubvae.eval.metrics.epoch_regression(
            folder,
            method,
            dataset_label,
            save_load=True,
            disentangle_keys=disentangle_keys,
        )

    if (method == "log_class") or ("_cv" in method):
        rows = 1
        figsize = (15, 10)
        metrics_keys = (
            ["Accuracy"] if ("log_class" in method) or ("qda" in method) else ["R2"]
        )
    else:
        rows = 2
        figsize = (15, 15)
        metrics_keys = ["R2", "R2_Null"]

    if task_id == "":
        ## Plot R^2
        for key in disentangle_keys:
            f, ax_arr = plt.subplots(rows, 1, figsize=figsize)
            plt.title("R2 of Regression of {} Using {}".format(key, method.title()))
            for path_i, p in enumerate(analysis_keys):

                for i, metric in enumerate(metrics_keys):
                    if rows == 1:
                        ax = ax_arr
                    else:
                        ax = ax_arr[i]

                    if "_cv" in method:
                        print(np.array(metrics[p][key][metric]).shape)
                        metric_to_plot = np.array(metrics[p][key][metric]).mean(axis=-1)
                    else:
                        metric_to_plot = metrics[p][key][metric]
                    argsort = np.argsort(metrics[p]["epochs"])
                    ax.plot(
                        np.array(metrics[p]["epochs"])[argsort],
                        np.array(metric_to_plot)[argsort],
                        label="{}".format(p),
                        color=palette[path_i],
                        alpha=0.5,
                    )

                    ax.set_ylabel(metric)
                    ax.legend()
                    ax.set_xlabel("Epoch")
                    print(
                        "{}{}{}:{}".format(p, key, metric, metrics[p][key][metric])
                    )  # max(min(metrics[p][key][metric])
                    # ax.set_ylim(bottom= 0)

                    # ax.set_ylim(top=1)

            f.tight_layout()
            plt.savefig("{}/{}_{}_epoch.png".format(out_path, key, method))
            plt.close()

elif method == "gmm_entropy":
    metrics = {}
    comparison_clustering = RESULTS_PATH + "vanilla/64/vis_latents/z_300_gmm.npy"
    for an_key in analysis_keys:
        folder = "{}/{}/".format(RESULTS_PATH, an_key)
        print("Reading in folder: {}".format(folder))
        metrics[an_key] = scrubvae.eval.metrics.epoch_cluster_entropy(
            path=folder,
            method=method,
            dataset_label=dataset_label,
            save_load=True,
            n_components=25,
            comparison_clustering=comparison_clustering,
        )

    if task_id == "":
        metric = "Entropy"
        f, ax = plt.subplots(1, 1, figsize=(15, 15))
        plt.title("GMM Clustering Entropy {}".format(method.title()))
        for path_i, p in enumerate(analysis_keys):

            if "_cv" in method:
                print(np.array(metrics[p][metric]).shape)
                metric_to_plot = np.array(metrics[p][metric]).mean(axis=-1)
            else:
                metric_to_plot = metrics[p][metric]

            argsort = np.argsort(metrics[p]["epochs"])
            ax.plot(
                np.array(metrics[p]["epochs"])[argsort],
                np.array(metric_to_plot)[argsort],
                label="{}".format(p),
                color=palette[path_i],
                alpha=0.5,
            )

            ax.set_ylabel(metric)
            ax.legend()
            ax.set_xlabel("Epoch")
            print(
                "{}{}:{}".format(p, metric, metrics[p][metric])
            )  # max(min(metrics[p][key][metric])
            # ax.set_ylim(bottom= 0)

            # ax.set_ylim(top=1)

        f.tight_layout()
        plt.savefig("{}/{}_epoch.png".format(out_path, method))
        plt.close()
