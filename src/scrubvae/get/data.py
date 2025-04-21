from neuroposelib import read
import numpy as np
from typing import List
import torch
from scrubvae.data.dataset import *
from scrubvae.data.dataset import preprocess_save_data
from torch.utils.data import DataLoader
import h5py
import pandas as pd


def mouse_data(
    data_config: dict,
    train_val_test: str = "train",
    data_keys: List[str] = ["x6d", "root", "offsets"],
    shuffle: bool = False,
    stride: int = 2,
    window: int = 51,
):
    """
    Load in mouse data and return pytorch dataloaders
    """
    skeleton_config = read.config(
        "{}mouse_skeleton.yaml".format(data_config["data_path"])
    )

    if train_val_test is not "full":
        data_path = "{}{}/{}/".format(
            data_config["data_path"], data_config["dataset"], train_val_test
        )
        data = {}
        for key in data_keys + ["ids"]:
            if key in ["pd_label", "fluorescence"]:
                continue
            elif key in ["ids", "heading", "avg_speed_3d", "offsets", "raw_pose"]:
                file_path = "{}{}.h5".format(data_path, key)
            else:
                file_path = "{}{}_{}.h5".format(
                    data_path, key, data_config["direction_process"]
                )
            print("Reading in {} from {}".format(key, file_path))
            hf = h5py.File(file_path, "r")
            data[key] = np.array(hf.get(key))
            hf.close()
        data = {k: torch.from_numpy(v) for k, v in data.items()}
    elif train_val_test == "full":
        data = preprocess_save_data(
            data_path=data_config["data_path"],
            skeleton_config=skeleton_config,
            dataset=data_config["dataset"],
            window=window,
            stride=stride,
            data_keys=data_keys + ["ids"],
            speed_threshold=2.25,
            direction_process=data_config["direction_process"],
        )

    norm_params = {
        "avg_speed_3d": {
            "mean": torch.tensor([0.4993, 0.7112, 0.6663], dtype=torch.float32),
            "std": torch.tensor([0.4038, 0.3586, 0.4169], dtype=torch.float32),
        }
    }
    if "avg_speed_3d" in data_keys:
        print("Mean centering and unit standard deviation-scaling avg_speed_3d")
        data["avg_speed_3d"] -= norm_params["avg_speed_3d"]["mean"]
        data["avg_speed_3d"] /= norm_params["avg_speed_3d"]["std"]

        print("Speed Mins and Maxes:")
        print(data["avg_speed_3d"].min(dim=0)[0])
        print(data["avg_speed_3d"].max(dim=0)[0])

    discrete_classes = {}
    if data_config["dataset"] == "parkinsons":
        # Only if read in raw poses for the PD dataset
        # if not ((data_config["stride"] == 5) or (data_config["stride"] == 10)):
        if "pd_label" in data_keys:
            data["pd_label"] = torch.zeros((len(data["ids"]), 1)).long()
            data["pd_label"][data["ids"] >= 36] = 1
            discrete_classes["pd_label"] = torch.unique(data["pd_label"], sorted=True)

        if "fluorescence" in data_keys:
            meta = pd.read_csv(
                data_config["data_path"] + data_config["dataset"] + "/metadata.csv"
            )
            # import pdb; pdb.set_trace()
            meta_by_frame = meta.iloc[data["ids"]]
            fluorescence = meta_by_frame["Fluorescence"].to_numpy()
            data["fluorescence"] = torch.tensor(fluorescence, dtype=torch.float32)

        data["ids"][data["ids"] >= 36] = data["ids"][data["ids"] >= 36] - 36
        unique_ids = torch.unique(data["ids"])
        discrete_classes["ids"] = torch.arange(len(unique_ids)).long()
    else:
        discrete_classes["ids"] = torch.unique(data["ids"], sorted=True)

    dataset = MouseDataset(
        data,
        data_config["arena_size"],
        skeleton_config["KINEMATIC_TREE"],
        len(skeleton_config["LABELS"]),
        label=train_val_test,
        discrete_classes=discrete_classes,
        norm_params=norm_params,
    )
    # if not train:
    #     # Read in validated clustering
    #     vanilla_path = "/mnt/home/jwu10/working/ceph/results/vae/vanilla_64/"
    #     gmm_dict = {
    #         "midfwd_train": [
    #             "{}2/vis_latents_train/z_300_gmm.npy".format(vanilla_path),
    #             [1, 4, 8, 38, 41, 44],
    #         ],
    #         "midfwd_test": [
    #             "{}test/vis_latents/z_600_gmm.npy".format(vanilla_path),
    #             [8, 23, 30, 34, 35, 37, 46],  # 23, 30
    #         ],
    #         "x360_test": [
    #             "{}test_x360/vis_latents/z_600_gmm.npy".format(vanilla_path),
    #             [],
    #         ],
    #     }

    #     dataset.gmm_pred = {"midfwd_test": np.load(gmm_dict["midfwd_test"][0])}
    #     dataset.walking_clusters = {k: v[1] for k, v in gmm_dict.items()}

    #     ## Pre-randomize avg speed to maintain overall distribution of speeds
    #     if "avg_speed_3d" in dataset.data.keys():
    #         dataset.data["avg_speed_3d_rand"] = dataset[:]["avg_speed_3d"][
    #             torch.randperm(
    #                 len(dataset), generator=torch.Generator().manual_seed(100)
    #             )
    #         ]

    #     if data_config["direction_process"]:
    #         dataset.gmm_pred["x360_test"] = np.load(gmm_dict["x360_test"][0])

    loader = DataLoader(
        dataset=dataset,
        batch_size=data_config["batch_size"],
        shuffle=shuffle,
        num_workers=5,
        pin_memory=True,
    )

    return loader
