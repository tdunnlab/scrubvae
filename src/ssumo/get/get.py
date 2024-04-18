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
    dataset, loader = ssumo.get.mouse_data(
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
        n_keypts=dataset.n_keypts,
        direction_process=config["data"]["direction_process"],
        arena_size=dataset.arena_size,
        kinematic_tree=dataset.kinematic_tree,
        bound=config["data"]["normalize"] == "bounded",
        device="cuda",
        verbose=verbose,
    )
    return dataset, loader, model


def all_saved_epochs(path):
    z_path = Path(path + "weights/")
    epochs = [re.findall(r"\d+", f.parts[-1]) for f in list(z_path.glob("epoch*"))]
    epochs = np.sort(np.array(epochs).astype(int).squeeze())
    print("Epochs found: {}".format(epochs))

    return epochs
