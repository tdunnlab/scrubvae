from base_path import RESULTS_PATH
import torch
import matplotlib.pyplot as plt
from ssumo.params import read
import ssumo
from tqdm import tqdm
from pathlib import Path
import numpy as np
import pickle
import sys


def plot_loss_curve(
    lossdict, times, dataset="Train", models=[], ylimit=False, plotkey="jpe", nametag=""
):
    if models == []:
        models = lossdict.keys()
    plt.figure(figsize=(10, 5))
    plt.title(dataset + " " + plotkey + " Loss over Time by Model")
    for model in models:
        yplot = [lossdict[model][epoch][plotkey] for epoch in sorted(lossdict[model])]
        plt.plot(
            np.linspace(0, len(yplot) * times[model], len(yplot)),
            # [times[model] for epoch in sorted(lossdict[model])],
            yplot,
            label=model,
            alpha=0.5,
            linewidth=1,
        )
    plt.yscale("log")
    if ylimit:
        plt.ylim(ylimit[0], ylimit[1])
    plt.xlabel("Time (Hours)")
    plt.ylabel("Log Loss")
    plt.legend()
    plt.savefig(
        "{}".format(RESULTS_PATH) + dataset + plotkey + "lossplot" + nametag + ".png"
    )
    plt.close()


task_id = sys.argv[1]
# can be a number (slurm array), model folder name to calculate losses, or "plot"

if task_id.isdigit():
    models = [f.parts[-1] for f in Path(RESULTS_PATH).iterdir() if f.is_dir()][
        int(task_id)
    ]
else:
    models = task_id

if task_id == "plot":
    models = [f.parts[-1] for f in Path(RESULTS_PATH).iterdir() if f.is_dir()]

    trainlossdict = {}
    testlossdict = {}
    # Train loss dictionaries
    for model in models:
        load_path = "{}/losses/TrainLosses.p".format(RESULTS_PATH + model)
        with open(load_path, "rb") as trainloss:
            loss_dict = pickle.load(trainloss)
        trainlossdict[model] = loss_dict
        load_path = "{}/losses/TestLosses.p".format(RESULTS_PATH + model)
        with open(load_path, "rb") as testloss:
            loss_dict = pickle.load(testloss)
        testlossdict[model] = loss_dict
    times = {
        model: np.average(
            [
                trainlossdict[model][epoch]["time"]
                for epoch in sorted(trainlossdict[model])
            ]
        )
        / 360
        for model in trainlossdict.keys()
    }
    plot_loss_curve(trainlossdict, times, "Train", models=models, ylimit=[70, 1000])
    plot_loss_curve(testlossdict, times, "Test", models=models)

else:
    # Train loss from dict
    load_path = "{}/losses/loss_dict.p".format(RESULTS_PATH + models)
    with open(load_path, "rb") as lossfile:
        loss_dict = pickle.load(lossfile)
    formattedtrainloss = [
        {key: loss_dict[key][i] for key in loss_dict}
        for i in range(len(loss_dict["total"]))
    ]
    trainlossdict = {
        epoch + 1: formattedtrainloss[epoch] for epoch in range(len(formattedtrainloss))
    }
    pickle.dump(
        trainlossdict,
        open("{}/losses/TrainLosses.p".format(RESULTS_PATH + models), "wb"),
    )

    # Test loss by running model
    config = read.config(RESULTS_PATH + models + "/model_config.yaml")
    ### Load Dataset
    dataset, loader = ssumo.data.get_mouse(
        data_config=config["data"],
        window=config["model"]["window"],
        train=False,
        data_keys=["x6d", "root", "offsets", "target_pose"]
        + config["disentangle"]["features"],
        shuffle=True,
    )

    testlosspath = "{}/losses/TestLosses.p".format(RESULTS_PATH + models)
    testlossdict = {}
    if Path(testlosspath).is_file():
        with open(testlosspath, "rb") as testloss:
            testlossdict = pickle.load(testloss)

    epochs_saved = ssumo.eval.metrics.get_all_epochs(RESULTS_PATH + models + "/")
    config = read.config(RESULTS_PATH + models + "/model_config.yaml")
    config["model"]["load_model"] = config["out_path"]

    for epoch in epochs_saved:
        if epoch in testlossdict.keys():
            continue
        config["model"]["start_epoch"] = epoch
        vae, device = ssumo.model.get(
            model_config=config["model"],
            disentangle_config=config["disentangle"],
            n_keypts=dataset.n_keypts,
            direction_process=config["data"]["direction_process"],
            arena_size=dataset.arena_size,
            kinematic_tree=dataset.kinematic_tree,
            verbose=1,
        )
        vae.to(device)
        epoch_loss = ssumo.train.train_epoch(
            vae,
            None,
            None,
            loader,
            device,
            config["loss"],
            epoch,
            mode="test",
            disentangle_keys=config["disentangle"]["features"],
        )
        testlossdict[epoch] = epoch_loss
    pickle.dump(
        testlossdict,
        open("{}/losses/TestLosses.p".format(RESULTS_PATH + models), "wb"),
    )
