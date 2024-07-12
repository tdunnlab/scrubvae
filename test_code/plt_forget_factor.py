import ssumo
from neuroposelib import read
import torch

CODE_PATH = "/mnt/home/jwu10/working/ssumo/"
RESULTS_PATH = "/mnt/ceph/users/jwu10/results/vae/"

paths = ["mals_updated/p2_20_b", "mals_64/mals_p1_20"]

ff = {}
for path in paths:
    epochs = ssumo.get.all_saved_epochs(RESULTS_PATH + path + "/")
    ff[path] = []
    for epoch in epochs:
        config = read.config(RESULTS_PATH + path + "/model_config.yaml")
        skeleton_config = read.config(config["data"]["skeleton_path"])

        model = ssumo.get.model(
                model_config=config["model"],
                load_model=RESULTS_PATH + path,
                epoch=epoch,
                disentangle_config=config["disentangle"],
                n_keypts=18,
                direction_process=config["data"]["direction_process"],
                loss_config=config["loss"],
                arena_size=torch.tensor(config["data"]["arena_size"]),
                kinematic_tree=skeleton_config["KINEMATIC_TREE"],
                bound=config["data"]["normalize"] == "bounded",
                discrete_classes = {},
                device="cpu",
                verbose=-1,
            )
        
        ff[path] += [model.disentangle["moving_avg_lsq"]["heading"].lam1.detach().numpy()[0]]

    print(ff)
        

