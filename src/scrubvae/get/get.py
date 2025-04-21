import scrubvae
import re
from pathlib import Path
import numpy as np


def data_and_model(
    config,
    load_model=None,
    epoch=None,
    train_val_test=["train", "val", "test"],
    data_keys=["x6d", "root", "offsets"],
    shuffle=False,
    use_default_val_keys=True,
    verbose=1,
):
    if use_default_val_keys:
        if config["data"]["dataset"] == "4_mice":
            val_data_keys = [
                "ids",
                "x6d",
                "root",
                "offsets",
                "target_pose",
                "avg_speed_3d",
                "heading",
            ]
        else:
            val_data_keys = [
                "ids",
                "x6d",
                "root",
                "offsets",
                "target_pose",
                "fluorescence",
                "pd_label",
            ]
    else:
        val_data_keys = data_keys

    if epoch is None:
        epoch = config["model"]["start_epoch"]

    if load_model is None:
        load_model = config["model"]["load_model"]

    ### Load Dataset
    # if train_val_test == "all":
    loader_dict = {}
    for is_shuffle, dataset_label in zip(shuffle, train_val_test):
        curr_data_keys = val_data_keys if dataset_label == "val" else data_keys
        loader_dict[dataset_label] = scrubvae.get.mouse_data(
            data_config=config["data"],
            train_val_test=dataset_label,
            data_keys=curr_data_keys,
            shuffle=is_shuffle,
            # normalize=["avg_speed_3d"] if "avg_speed_3d" in curr_data_keys else None,
            # norm_params=None,
        )
        
        # import pdb; pdb.set_trace()
        # loader2 = scrubvae.get.mouse_data(
        #     data_config=test_config,
        #     window=config["model"]["window"],
        #     train=False,
        #     data_keys=[
        #         "x6d",
        #         "root",
        #         "offsets",
        #         "target_pose",
        #         "avg_speed_3d",
        #         "heading",
        #     ],
        #     shuffle=False,
        #     normalize=["avg_speed_3d"],
        #     norm_params=loader1.dataset.norm_params,
        # )
    # else:
    #     loader1 = scrubvae.get.mouse_data(
    #         data_config=config["data"],
    #         window=config["model"]["window"],
    #         train=train_val_test,
    #         data_keys=data_keys,
    #         shuffle=shuffle,
    #         normalize=config["disentangle"]["features"],
    #     )
    model = scrubvae.get.model(
        model_config=config["model"],
        load_model=load_model,
        epoch=epoch,
        disentangle_config=config["disentangle"],
        n_keypts=loader_dict[train_val_test[0]].dataset.n_keypts,
        direction_process=config["data"]["direction_process"],
        loss_config=config["loss"],
        arena_size=loader_dict[train_val_test[0]].dataset.arena_size,
        kinematic_tree=loader_dict[train_val_test[0]].dataset.kinematic_tree,
        bound=config["data"]["normalize"] == "bounded",
        discrete_classes=loader_dict[train_val_test[0]].dataset.discrete_classes,
        device="cuda",
        verbose=verbose,
    )

    return loader_dict, model


def all_saved_epochs(path):
    z_path = Path(path + "weights/")
    epochs = [re.findall(r"\d+", f.parts[-1]) for f in list(z_path.glob("epoch*"))]
    epochs = np.sort(np.array(epochs).astype(int).squeeze())
    print("Epochs found: {}".format(epochs))

    return epochs
