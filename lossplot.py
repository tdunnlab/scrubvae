from base_path import RESULTS_PATH
import torch
import matplotlib.pyplot as plt
from ssumo.params import read
import ssumo
import pdb
from tqdm import tqdm
import os
import numpy as np
import pickle

trainlosslist = []
testlosslist = []
averagetimes = [446, 728, 336, 485, 800, 1121]
models = [
    "n_layer/2_layer",
    "n_layer/3_layer",
    "straightlayers/ch64",
    "straightlayers/ch128",
    "straightlayers/ch256",
    "no_bias",
]
# models = ["no_bias"]

for j in range(len(models)):
    i = 10
    run = models[j]
    load_path = "{}/losses/loss_dict.pth".format(RESULTS_PATH + run)
    # loss_dict = torch.load(load_path)
    with open(load_path, "rb") as lossfile:
        loss_dict = pickle.load(lossfile)
    trainlosslist.append(loss_dict)

pickle.dump(trainlosslist, open("{}TrainLosses.pkl".format(RESULTS_PATH), "wb"))

config = read.config(RESULTS_PATH + run + "/model_config.yaml")
### Load Dataset
dataset, loader = ssumo.data.get_mouse(
    data_config=config["data"],
    window=config["model"]["window"],
    train=False,
    data_keys=["x6d", "root", "offsets"] + config["disentangle"]["features"],
    shuffle=True,
)
device = "cuda:0" if torch.cuda.is_available() else "cpu"

for j in tqdm(range(len(models))):
    print(j)
    i = 10
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
    while True:
        load_path = "{}/weights/epoch_{}.pth".format(RESULTS_PATH + run, i)
        if os.path.isfile(load_path):
            state_dict = torch.load(load_path)
        else:
            break
        # state_dict["arena_size"] = arena_size.cuda()
        # pdb.set_trace()
        vae.load_state_dict(state_dict)
        vae.to(device)
        epoch_loss = ssumo.train.train_epoch(
            vae,
            "optimizer",
            loader,
            device,
            config["loss"],
            i,
            mode="test",
            disentangle_keys=config["disentangle"]["features"],
        )
        loss.append(epoch_loss)
        i += 10
    testlosslist.append(loss)
    pickle.dump(testlosslist, open("{}TestLossesnobias.pkl".format(RESULTS_PATH), "wb"))


with open("{}TrainLosses.pkl".format(RESULTS_PATH), "rb") as lossfile:
    trainlosslist = pickle.load(lossfile)
with open("{}TestLosses.pkl".format(RESULTS_PATH), "rb") as lossfile:
    testlosslist = pickle.load(lossfile)


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
plt.ylim(100, 1000)
plt.yscale("log")
plt.xlabel("Time (minutes)")
plt.ylabel("Log Loss")
plt.legend()
plt.savefig("{}TrainJPElossplot.png".format(RESULTS_PATH))
plt.close()


plt.figure(2)
plt.title("Test JPE Loss over Time by Model")
for i in range(len(testlosslist) - 1):
    # pdb.set_trace()
    max_epoch = len(testlosslist[i])
    plt.plot(
        np.linspace(0, max_epoch * averagetimes[i] * 10 / 60, max_epoch + 1)[1:],
        [j["jpe"] for j in testlosslist[i]],
        label=models[i],
        alpha=0.5,
        linewidth=1,
    )

# with open("{}TestLossesnobias.pkl".format(RESULTS_PATH), "rb") as lossfile:
#     nobiastestloss = pickle.load(lossfile)

# # pdb.set_trace()

# max_epoch = len(nobiastestloss[0])
# plt.plot(
#     np.linspace(0, max_epoch * averagetimes[5] * 10 / 60, max_epoch + 1)[1:],
#     [j["jpe"] for j in nobiastestloss[0]],
#     label=models[5],
#     alpha=0.5,
#     linewidth=1,
# )
plt.yscale("log")
plt.xlabel("Time (minutes)")
plt.ylabel("Log Loss")
plt.legend()
plt.savefig("{}TestJPElossplot.png".format(RESULTS_PATH))
plt.close()
