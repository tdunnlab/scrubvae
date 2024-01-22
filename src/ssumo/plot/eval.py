from ssumo.plot.constants import PALETTE_DICT
import matplotlib.pyplot as plt
import numpy as np


def loss(loss_dict, out_path, disentangle_keys=None):
    if any(["_gr" in k for k in loss_dict.keys()]):
        disentangle_loss_keys = disentangle_keys + [k + "_gr" for k in disentangle_keys]
    else:
        disentangle_loss_keys = disentangle_keys
    vae_loss_keys = [k for k in loss_dict.keys() if k not in disentangle_loss_keys]

    for keys in [("vae", vae_loss_keys), ("disentangle", disentangle_loss_keys)]:
        if keys[1] is not None:
            f = plt.figure(figsize=(15, 10))
            for i, k in enumerate(keys[1]):
                plt.plot(
                    np.arange(1, len(loss_dict[k])),
                    loss_dict[k][:-1],
                    label=k,
                    c=PALETTE_DICT[i],
                )

            plt.yscale("log")
            plt.xlabel("Epoch")
            plt.ylabel("Log Loss")
            plt.legend()
            plt.savefig("{}/losses/{}.png".format(out_path, keys[0]))
            plt.close()
