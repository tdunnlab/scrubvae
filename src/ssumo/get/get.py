import ssumo
import re
from pathlib import Path
import numpy as np


def data_and_model(
    config,
    load_model=None,
    epoch=None,
    dataset_label="Train",
    data_keys=["x6d", "root", "offsets"],
    shuffle=False,
    verbose=1,
):
    if epoch is None:
        epoch = config["model"]["start_epoch"]

    if load_model is None:
        load_model = config["model"]["load_model"]

    ### Load Dataset
    if dataset_label == "Both":
        loader1 = ssumo.get.mouse_data(
            data_config=config["data"],
            window=config["model"]["window"],
            train=True,
            data_keys=data_keys,
            shuffle=shuffle,
            normalize=config["disentangle"]["features"],
            norm_params=None,
        )
        test_config = config["data"]
        test_config["stride"] = 1
        loader2 = ssumo.get.mouse_data(
            data_config=test_config,
            window=config["model"]["window"],
            train=False,
            data_keys=[
                "x6d",
                "root",
                "offsets",
                "target_pose",
                "avg_speed_3d",
                "heading",
            ],
            shuffle=False,
            normalize=["avg_speed_3d"],
            norm_params=loader1.dataset.norm_params,
        )
    else:
        loader1 = ssumo.get.mouse_data(
            data_config=config["data"],
            window=config["model"]["window"],
            train=dataset_label == "Train",
            data_keys=data_keys,
            shuffle=shuffle,
            normalize=config["disentangle"]["features"],
        )

    model = ssumo.get.model(
        model_config=config["model"],
        load_model=load_model,
        epoch=epoch,
        disentangle_config=config["disentangle"],
        n_keypts=loader1.dataset.n_keypts,
        direction_process=config["data"]["direction_process"],
        loss_config=config["loss"],
        arena_size=loader1.dataset.arena_size,
        kinematic_tree=loader1.dataset.kinematic_tree,
        bound=config["data"]["normalize"] == "bounded",
        discrete_classes=loader1.dataset.discrete_classes,
        device="cuda",
        verbose=verbose,
    )

    if dataset_label == "Both":
        return loader1, loader2, model
    else:
        return loader1, model


def all_saved_epochs(path):
    z_path = Path(path + "weights/")
    epochs = [re.findall(r"\d+", f.parts[-1]) for f in list(z_path.glob("epoch*"))]
    epochs = np.sort(np.array(epochs).astype(int).squeeze())
    print("Epochs found: {}".format(epochs))

    return epochs
