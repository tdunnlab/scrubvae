import numpy as np
import matplotlib.pyplot as plt
import ssumo
from base_path import RESULTS_PATH
import sys
from pathlib import Path

palette = ssumo.plot.constants.PALETTE_2
experiment_folder = sys.argv[1]
results_path = Path(RESULTS_PATH + experiment_folder + "/")

if len(sys.argv) > 2:
    task_id = sys.argv[2]
else:
    task_id = None

if task_id is None:
    analysis_keys = [f.parts[-1] for f in results_path.iterdir() if f.is_dir()]
elif task_id.isdigit():
    analysis_keys = [
        [f.parts[-1] for f in results_path.iterdir() if f.is_dir()][int(task_id)]
    ]
else:
    analysis_keys = [task_id]

dataset_label = "Train"
disentangle_keys = ["avg_speed_3d", "heading"]

metrics = {}
for an_key in analysis_keys:
    folder = "{}/{}/".format(results_path, an_key)
    print("Reading in folder: {}".format(folder))
    metrics[an_key] = ssumo.eval.metrics.epoch_adversarial_attack(
        folder, dataset_label, save_load=True, disentangle_keys = disentangle_keys
    )

if task_id is None:
    ## Plot R^2
    for key in disentangle_keys:
        f, ax_arr = plt.subplots(2, 1, figsize=(15, 15))
        for path_i, p in enumerate(analysis_keys):
            for i, metric in enumerate(["R2", "R2_Null"]):
                if "Norm" in metric:
                    ax_arr[i].plot(
                        metrics[p]["epochs"],
                        np.log10(metrics[p][key][metric]),
                        label="{}".format(p),
                    )
                else:
                    ax_arr[i].plot(
                        metrics[p]["epochs"],
                        metrics[p][key][metric],
                        label="{}".format(p),
                        color=palette[path_i],
                        alpha=0.5,
                    )

                ax_arr[i].set_ylabel(metric)
                ax_arr[i].legend()
                ax_arr[i].set_xlabel("Epoch")
                ax_arr[i].set_ylim(bottom=max(min(metrics[p][key][metric]), 0))

                ax_arr[i].set_ylim(bottom=0, top=1)

        f.tight_layout()
        plt.savefig("{}/{}_adv_atk_epoch.png".format(results_path, key))
        plt.close()
