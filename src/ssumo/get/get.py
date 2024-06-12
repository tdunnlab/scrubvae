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
    is_2D=False,
):
    if epoch is None:
        epoch = config["model"]["start_epoch"]

    if load_model is None:
        load_model = config["model"]["load_model"]

    ### Load Dataset
    loader = ssumo.get.mouse_data(
        data_config=config["data"],
        window=config["model"]["window"],
        train=dataset_label == "Train",
        data_keys=data_keys,
        shuffle=shuffle,
        normalize=config["disentangle"]["features"],
        is_2D=is_2D,
    )

    model = ssumo.get.model(
        model_config=config["model"],
        load_model=load_model,
        epoch=epoch,
        disentangle_config=config["disentangle"],
        n_keypts=loader.dataset.n_keypts,
        direction_process=config["data"]["direction_process"],
        loss_config=config["loss"],
        arena_size=loader.dataset.arena_size,
        kinematic_tree=loader.dataset.kinematic_tree,
        bound=config["data"]["normalize"] == "bounded",
        discrete_classes=loader.dataset.discrete_classes,
        device="cuda",
        verbose=verbose,
    )
    return loader, model


def all_saved_epochs(path):
    z_path = Path(path + "weights/")
    epochs = [re.findall(r"\d+", f.parts[-1]) for f in list(z_path.glob("epoch*"))]
    epochs = np.sort(np.array(epochs).astype(int).squeeze())
    print("Epochs found: {}".format(epochs))

    return epochs
