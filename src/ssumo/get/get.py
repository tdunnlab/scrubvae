from ssumo import get
import re
from pathlib import Path
import numpy as np

def data_and_model(config, epoch, dataset_label="Train", shuffle=False):

    ### Load Dataset
    dataset, loader = get.mouse_data(
        data_config=config["data"],
        window=config["model"]["window"],
        train=dataset_label=="Train",
        data_keys=["x6d", "root", "offsets", "target_pose"]
        + config["disentangle"]["features"],
        shuffle=shuffle,
        normalize=config["disentangle"]["features"],
    )

    model = get.model(
        model_config=config["model"],
        epoch = epoch,
        disentangle_config=config["disentangle"],
        n_keypts=dataset.n_keypts,
        direction_process=config["data"]["direction_process"],
        arena_size=dataset.arena_size,
        kinematic_tree=dataset.kinematic_tree,
        bound=config["data"]["normalize"] is not None,
        device="cuda",
        verbose=1,
    )
    return dataset, loader, model


def all_saved_epochs(path):
    z_path = Path(path + "weights/")
    epochs = [re.findall(r"\d+", f.parts[-1]) for f in list(z_path.glob("epoch*"))]
    epochs = np.sort(np.array(epochs).astype(int).squeeze())
    print("Epochs found: {}".format(epochs))

    return epochs