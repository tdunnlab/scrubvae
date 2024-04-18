import ssumo
from dappy import read
from base_path import RESULTS_PATH
import sys
import numpy as np
from sklearn.linear_model import LogisticRegression


analysis_key = sys.argv[1]
epoch = 560
config = read.config(RESULTS_PATH + analysis_key + "/model_config.yaml")
dataset_label = "Train"
dataset, loader = ssumo.get.mouse_data(
    config["data"],
    window=config["model"]["window"],
    train=dataset_label == "Train",
    data_keys=["ids"],
    shuffle=False,
    normalize=config["disentangle"]["features"],
)

latents = ssumo.get.latents(
    config, epoch=epoch, device="cuda", dataset_label=dataset_label
)

LR_Classifier = LogisticRegression(multi_class="ovr")
LR_Classifier.fit(latents, dataset[:]["ids"][:,0])
preds = LR_Classifier.predict(latents)




