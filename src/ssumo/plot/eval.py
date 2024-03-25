from ssumo.plot.constants import PALETTE_DICT
import matplotlib.pyplot as plt
import numpy as np


def loss(loss_dict, out_path, disentangle_keys=[]):
    gr_keys = [k + "_gr" for k in disentangle_keys]
    disentangle_loss_keys = [k for k in loss_dict.keys() if k in (disentangle_keys + gr_keys)]
    vae_loss_keys = [k for k in loss_dict.keys() if k not in disentangle_loss_keys + ["mcmi"]]
    plot_tuples = [("vae", vae_loss_keys)]
    plot_tuples += [("disentangle", disentangle_loss_keys)] if len(disentangle_loss_keys)>0 else []
    plot_tuples += [("mcmi", ["mcmi"])] if "mcmi" in loss_dict.keys() else []
    for keys in plot_tuples:
        if len(keys[1])>0:
            f = plt.figure(figsize=(15, 10))
            for i, k in enumerate(keys[1]):
                plt.plot(
                    np.arange(1, len(loss_dict[k])),
                    loss_dict[k][:-1],
                    label=k,
                    c=PALETTE_DICT[i],
                )

            if keys[0] != "mcmi":
                plt.yscale("log")
            plt.xlabel("Epoch")
            plt.ylabel("Log Loss")
            plt.legend()
            plt.savefig("{}/losses/{}.png".format(out_path, keys[0]))
            plt.close()
