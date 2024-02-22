from ssumo.data.dataset import fwd_kin_cont6d_torch

from torch.utils.data import DataLoader
from dappy import read
import torch
from dappy import vis
import ssumo
from base_path import RESULTS_PATH
import sys
import matplotlib.pyplot as plt
from ssumo.params import read
import pdb
from tqdm import tqdm
import os
import numpy as np
import pickle

analysis_key = sys.argv[1]
if sys.argv[1] != "lossplot":
    config = read.config(RESULTS_PATH + analysis_key + "/model_config.yaml")
    config["model"]["start_epoch"] = sys.argv[2]
    config["model"]["load_model"] = config["out_path"]
    config["data"]["stride"] = 10
    config["data"]["batch_size"] = 10

    connectivity = read.connectivity_config(config["data"]["skeleton_path"])

    ### Load Datasets
    train_dataset, train_loader = ssumo.data.get_mouse(
        data_config=config["data"],
        window=config["model"]["window"],
        train=True,
        data_keys=["x6d", "root", "offsets", "raw_pose"],
        shuffle=True,
    )

    test_dataset, test_loader = ssumo.data.get_mouse(
        data_config=config["data"],
        window=config["model"]["window"],
        train=False,
        data_keys=["x6d", "root", "offsets", "raw_pose"],
        shuffle=True,
    )

    vae, device = ssumo.model.get(
        model_config=config["model"],
        disentangle_config=config["disentangle"],
        n_keypts=train_dataset.n_keypts,
        direction_process=config["data"]["direction_process"],
        arena_size=train_dataset.arena_size,
        kinematic_tree=train_dataset.kinematic_tree,
        verbose=1,
    )
    kinematic_tree = train_dataset.kinematic_tree
    n_keypts = train_dataset.n_keypts

    # vae = utils.init_model(config, n_keypts, config["invariant"]).cuda()
    # arena_size = None if config["arena_size"] is None else train_dataset.arena_size.cuda()

    def visualize_reconstruction(loader, label):
        vae.eval()
        with torch.no_grad():
            # Let's see how reconstruction looks on train data
            data = next(iter(loader))
            data = {k: v.to(device) for k, v in data.items()}
            data_o = ssumo.train.predict_batch(
                vae, data, disentangle_keys=config["disentangle"]["features"]
            )

            pose = fwd_kin_cont6d_torch(
                data["x6d"].reshape(-1, n_keypts, 6),
                kinematic_tree,
                data["offsets"].view(-1, n_keypts, 3),
                data["root"].reshape(-1, 3),
                do_root_R=True,
            )

            pose_hat = fwd_kin_cont6d_torch(
                data_o["x6d"].reshape(-1, n_keypts, 6),
                kinematic_tree,
                data["offsets"].view(-1, n_keypts, 3),
                data_o["root"].reshape(-1, 3),
                do_root_R=True,
            )

            pose_array = torch.cat(
                [data["raw_pose"].reshape(-1, n_keypts, 3), pose, pose_hat], axis=0
            )

            vis.pose.grid3D(
                pose_array.cpu().detach().numpy(),
                connectivity,
                frames=[
                    0,
                    config["data"]["batch_size"] * config["model"]["window"],
                    2 * config["data"]["batch_size"] * config["model"]["window"],
                ],
                centered=False,
                subtitles=["Raw", "Raw -> 6D -> Back", "VAE Reconstructed"],
                title=label + " Data",
                fps=45,
                figsize=(36, 12),
                N_FRAMES=config["data"]["batch_size"] * config["model"]["window"],
                VID_NAME=label + ".mp4",
                SAVE_ROOT=config["out_path"],
            )

    visualize_reconstruction(train_loader, "Train")
    visualize_reconstruction(test_loader, "Test")
else:
    ## TODO: Have average time be something that is saved during training so you don't have to hardcode it as an input here
    averagetimes = [446, 728, 336, 485, 800, 1121, 1094, 1138, 1000, 1000]
    ## TODO: Submit each of these as a job array. Reference `vis_epoch_regression.py`
    models = [
        "n_layer/2_layer",
        "n_layer/3_layer",
        "straightlayers/ch64",
        "straightlayers/ch128",
        "straightlayers/ch256",
        "vanilla",
        "no_rotation_loss",
        "latentdim/l32",
        "latentdim/l128",
        "no_bias",
    ]

    trainlosslist = []
    ## TODO: If "lossplot/" is a hardcoded string, it should be in the string portion, not the `.format()` portion.
    ## TODO: I prefer using `.p` instead of `.pkl`. Change to make consistent.
    ## TODO: Use the already existing `losses` folder in each model folder to save and load 
    ## TODO: No need to create a separate `(Test or Train)Losses.pkl` file which contains all of the losses models. This seems like file bloat. Just load in every time from model folders. Should be trivial slowdown.
    with open("{}TestLosses.pkl".format(RESULTS_PATH + "lossplot/"), "rb") as lossfile:
        testlosslist = pickle.load(lossfile) ## TODO: testlosslist doesn't look like it's a list.
    # testlosslist = {}
    ## TODO: Implementation of test and train should be standardized so that the same code works for reading/loading both.
    ## TODO (cont.): Ideally you have a variable `dataset_label` that you can set to "Train" or "Test" and it will plot/calculate losses for the appropriate dataset.
    # Train loss dictionaries
    for j in range(len(models)):
        load_path = "{}/losses/loss_dict.pth".format(RESULTS_PATH + models[j])
        # loss_dict = torch.load(load_path)
        with open(load_path, "rb") as lossfile:
            loss_dict = pickle.load(lossfile)
        trainlosslist.append(loss_dict)

    pickle.dump(
        trainlosslist,
        open("{}TrainLosses.pkl".format(RESULTS_PATH + "lossplot/"), "wb"),
    )

    # Test loss by running model
    config = read.config(RESULTS_PATH + models[0] + "/model_config.yaml")
    ### Load Dataset
    dataset, loader = ssumo.data.get_mouse(
        data_config=config["data"],
        window=config["model"]["window"],
        train=False,
        data_keys=["x6d", "root", "offsets", "target_pose"]
        + config["disentangle"]["features"],
        shuffle=True,
    )
    ## TODO: `device` is already set later when loading in the model. This is redundant
    device = "cuda:0" if torch.cuda.is_available() else "cpu"

    ##TODO: This should probably be enumerate. Don't use `j`. It's an uninformative name.
    for j in tqdm(range(len(models))):
        ## TODO: `i` seems to be a hard coded counter variable which loads in files based on epoch? This is unnecessary. 
        ## Instead use pathlib to look into the `/weights/` folder and load in the list of available saved models. And compare to your test_losses dictionary.
        ## There's some code in `ssumo.eval.metrics` which shows how to do this. It's used in `vis_epoch_regression.py`
        i = 10 ##TODO: Don't call this i. It's an uninformative name. I'm guessing this should be `epoch`?
        if models[j] in testlosslist:
            i = len(testlosslist[models[j]]) * 10 + 10
        else:
            testlosslist[models[j]] = []
        run = models[j]
        print(run)
        loss = []
        config = read.config(RESULTS_PATH + run + "/model_config.yaml")
        vae, device = ssumo.model.get(
            model_config=config["model"],
            disentangle_config=config["disentangle"],
            n_keypts=dataset.n_keypts,
            direction_process=config["data"]["direction_process"],
            arena_size=dataset.arena_size,
            kinematic_tree=dataset.kinematic_tree,
            verbose=1,
        )
        ## TODO: Don't like while loops. They make too many assumptions. Change this to a for loop.
        ## Won't need while loop if you use pathlib to pre-load in all filenames in a folder.
        ## TODO: use `pathlib` instead of `os.path`. There are many examples throughout ssumo of how to use `pathlib`.
        while os.path.isfile("{}/weights/epoch_{}.pth".format(RESULTS_PATH + run, i)):
            print("{}/weights/epoch_{}.pth".format(RESULTS_PATH + run, i))
            ## TODO: Loading in a model is already handled by `ssumo.model.get`
            ## Just set `config["load_model"] = config["out_path"]`
            ## And `config["start_epoch"] = curr_epoch` before `ssumo.model.get`
            state_dict = torch.load(
                "{}/weights/epoch_{}.pth".format(RESULTS_PATH + run, i)
            )
            vae.load_state_dict(state_dict)
            vae.to(device)
            epoch_loss = ssumo.train.train_epoch(
                vae,
                "optimizer", ## Why is optimizer a string?
                None,
                loader,
                device,
                config["loss"],
                i,
                mode="test",
                disentangle_keys=config["disentangle"]["features"],
            )
            loss.append(epoch_loss)
            i += 10
        testlosslist[run].append(loss)
        pickle.dump( ## TODO: dump this in the model `/losses/` folder.
            testlosslist,
            open("{}TestLosses.pkl".format(RESULTS_PATH + "lossplot/"), "wb"),
        )

    plt.figure(figsize=(10, 5))
    plt.title("Train JPE Loss over Time by Model")
    for i in range(len(trainlosslist)):
        max_epoch = len(trainlosslist[i]["jpe"])
        plt.plot(
            np.linspace(0, max_epoch * averagetimes[i] / 60, max_epoch + 1)[1:],
            trainlosslist[i]["jpe"],
            label=models[i],
            alpha=0.5,
            linewidth=1,
        )
    plt.ylim(70, 1000)
    plt.yscale("log")
    plt.xlabel("Time (minutes)")
    plt.ylabel("Log Loss")
    plt.legend()
    plt.savefig("{}TrainJPElossplot.png".format(RESULTS_PATH))
    plt.close()

    ## TODO: This plotting code looks identical to the code above? Why not make a function?
    plt.figure(figsize=(10, 5))
    plt.title("Test JPE Loss over Time by Model")
    for i in testlosslist.keys(): # TODO: for key in testlosslist.keys(): ## Or you can use k
        max_epoch = len(testlosslist[i])
        plt.plot(
            np.linspace(
                0, max_epoch * averagetimes[models.index(i)] * 10 / 60, max_epoch + 1
            )[1:],
            [j["jpe"] for j in testlosslist[i]], # TODO: don't use `j`.
            label=i,
            alpha=0.5,
            linewidth=1,
        )
    plt.yscale("log")
    plt.xlabel("Time (minutes)")
    plt.ylabel("Log Loss")
    plt.legend()
    plt.savefig("{}TestJPElossplot.png".format(RESULTS_PATH))
    plt.close()
