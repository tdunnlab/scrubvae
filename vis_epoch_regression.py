import numpy as np
import matplotlib.pyplot as plt
import ssumo
from base_path import RESULTS_PATH

palette = ssumo.plot.constants.PALETTE_2
results_path = RESULTS_PATH + "/fine_tune/"
paths = ["adamw_bal_var","adamw_cawr_bal_var","adamw_cawr_big_scrub","vanilla"]
# paths = [
#     "vanilla",
#     "gre1_b1_true_x360",
#     "balanced",
#     "no_gr",
#     "bal_hc_sum",
#     "adamw",
#     "adamw_big_scrub",
#     "adamw_cawr"
#     "bal_var_cawr",
# ]
# paths = ["4", "6", "8", "10"]
dataset_label = "Train"

disentangle_keys = ["avg_speed", "heading", "heading_change"]
metrics = {}
for path_ind, path in enumerate(paths):
    metrics[path] = ssumo.eval.metrics.epoch_adversarial_attack(
        "{}{}/".format(results_path, path), dataset_label, save_load=True
    )

## Plot R^2
for key in disentangle_keys:
    f, ax_arr = plt.subplots(2, 1, figsize=(15, 15))
    for path_i, p in enumerate(paths):
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

    f.tight_layout()
    plt.savefig(results_path + "/{}_adv_atk_epoch.png".format(key))
    plt.close()
