import numpy as np
import re
from pathlib import Path
from dappy import read
from ..data import get_mouse
from ..model import get
from .get import latents
from . import project_to_null
from sklearn.metrics import r2_score
from sklearn.linear_model import LinearRegression

def get_all_epochs(path):
    z_path = Path(path + "weights/")
    epochs = [re.findall(r"\d+", f.parts[-1]) for f in list(z_path.glob("epoch*"))]
    epochs = np.sort(np.array(epochs).astype(int).squeeze())
    print("Epochs found: {}".format(epochs))

    return epochs

def epoch_linear_regression(path, dataset_label = "Train", save=True):
    config = read.config(path + "/model_config.yaml")
    config["model"]["load_model"] = config["out_path"]

    disentangle_keys = config["disentangle"]["features"]
    dataset = get_mouse(
        data_config=config["data"],
        window=config["model"]["window"],
        train=dataset_label == "Train",
        data_keys=[
            "x6d",
            "root",
        ]
        + disentangle_keys,
        shuffle = False,
    )[0]

    epochs = get_all_epochs(path)
    metrics = {k: {"R2": [], "R2_Null": []} for k in disentangle_keys}
    for epoch_ind, epoch in enumerate(epochs):
        config["model"]["start_epoch"] = epoch

        vae, device = get(
            model_config=config["model"],
            disentangle_config=config["disentangle"],
            n_keypts=dataset.n_keypts,
            direction_process=config["data"]["direction_process"],
            arena_size=dataset.arena_size,
            kinematic_tree=dataset.kinematic_tree,
            verbose=-1,
        )

        z = latents(vae, dataset, config, device, dataset_label)

        for key in disentangle_keys:
            print("Decoding Feature: {}".format(key))
            y_true = dataset[:][key].detach().cpu().numpy()
            lin_model = LinearRegression().fit(z, y_true)
            pred = lin_model.predict(z)
            print(metrics)

            metrics[path][key]["R2"] += [r2_score(y_true, pred)]
            print(metrics[path][key]["R2"])

            if len(vae.disentangle.keys()) > 0:
                dis_w = vae.disentangle[key].decoder.weight.detach().cpu().numpy()
            else:
                dis_w = lin_model.coef_
                # z -= lin_model.intercept_[:,None] * dis_w

            ## Null space projection
            z_null = project_to_null(z, dis_w)[0]
            pred_null = LinearRegression().fit(z_null, y_true).predict(z_null)

            metrics[path][key]["R2_Null"] += [r2_score(y_true, pred_null)]
            print(metrics[path][key]["R2_Null"])

    if save:
        pickle.dump(metrics, open("{}/linreg.p".format(results_path), "wb"))

    return metrics
