import ssumo
import torch
from ssumo.params import read
import sys
from base_path import RESULTS_PATH

### Set/Load Parameters
analysis_key = sys.argv[1]

if len(sys.argv) > 2:
    job_id = sys.argv[2]
    print(job_id)
    print(sys.argv)
    analysis_key = "{}/{}/".format(analysis_key, job_id)

config = read.config(RESULTS_PATH + analysis_key + "/model_config.yaml")

dataset, loader, model = ssumo.get.data_and_model(
    config,
    dataset_label="Train",
    data_keys=["x6d", "root", "offsets", "target_pose"]
    + config["disentangle"]["features"],
    shuffle=True,
)

model = ssumo.train.train(model,loader)