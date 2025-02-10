from neuroposelib import read, preprocess
import numpy as np
import scrubvae.data.quaternion as qtn
from typing import List
import torch
from scrubvae.data.dataset import *
from torch.utils.data import DataLoader
from pathlib import Path
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt

data_path = "/mnt/home/jwu10/working/ceph/data/ensemble_healthy/pose_aligned.h5"
REORDER = [4, 3, 2, 1, 0, 5, 11, 10, 9, 8, 7, 6, 17, 16, 15, 14, 13, 12]

pose, ids = read.pose_h5(data_path, dtype=np.float64)
pose = pose[:, REORDER, :]
n_ids = len(np.unique(ids))
window = 51
window_inds = get_window_indices(ids, 1, window).reshape(n_ids, -1, window)
pw_inds = np.concatenate(
    [
        np.zeros((n_ids, window // 2, window), dtype=int),
        window_inds,
        np.zeros((n_ids, window - window // 2, window), dtype=int),
    ],
    axis=1,
)
pw_inds = pw_inds.reshape(n_ids, 3, 20, -1, window)
pw_inds = pw_inds[..., window//2:-window//2, :] #(ids, section, min, window, frame)

pose = preprocess.median_filter(pose, ids, 5)

def get_random():
    train, val, test = [], [], []
    for i in range(n_ids):
        for j in range(3):
            rand_inds = np.random.permutation(np.arange(20))
            print(rand_inds)
            train += [pw_inds[i, j, rand_inds[:10], :]]
            val += [pw_inds[i, j, rand_inds[10:15], :]]
            test += [pw_inds[i, j, rand_inds[15:], :]]

    train = np.concatenate(train, axis=0).reshape(-1, window)
    val = np.concatenate(val, axis=0).reshape(-1, window)
    test = np.concatenate(test, axis=0).reshape(-1, window)

    train = train[np.argsort(train[:,0]),:]
    val = val[np.argsort(val[:,0]),:]
    test = test[np.argsort(test[:,0]),:]

    assert len(np.intersect1d(train.flatten(), val.flatten())) == 0
    assert len(np.intersect1d(test.flatten(), val.flatten())) == 0
    assert len(np.intersect1d(train.flatten(), test.flatten())) == 0

    pose_train = pose[train]
    pose_val = pose[val]
    pose_test = pose[test]

    f, ax_arr = plt.subplots(3, 1)
    for i, pose_arr in enumerate([pose_train, pose_val, pose_test]):
        speed = np.diff(pose_arr, n=1, axis=1)
        speed = np.sqrt((speed**2).sum(axis=-1)).mean(axis=-1)
        ax_arr[i].hist(speed, bins=100, range = (0, 3))

    plt.savefig("/mnt/home/jwu10/working/ceph/data/ensemble_healthy/speed_hist.png")
    plt.close()

    f, ax_arr = plt.subplots(3, 1)
    for i, pose_arr in enumerate([pose_train, pose_val, pose_test]):
        yaw = get_frame_yaw(pose_arr[:, window//2, ...], 0, 1)
        ax_arr[i].hist(yaw, bins=100)

    plt.savefig("/mnt/home/jwu10/working/ceph/data/ensemble_healthy/yaw_hist.png")
    plt.close()

    return pose_train, pose_val, pose_test

pose_train, pose_val, pose_test = get_random()
import pdb; pdb.set_trace()
np.save("/mnt/home/jwu10/working/ceph/data/wu_iclr25/4_mice/train.npy", pose_train)
np.save("/mnt/home/jwu10/working/ceph/data/wu_iclr25/4_mice/validation.npy", pose_val)
np.save("/mnt/home/jwu10/working/ceph/data/wu_iclr25/4_mice/test.npy", pose_test)