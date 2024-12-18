import scrubbed_cvae
from scrubbed_cvae.params import read
from scripts.base_path import RESULTS_PATH
from pathlib import Path
import wandb
import argparse

# argparse project and job names
parser = argparse.ArgumentParser(prog="SC-VAE Train", description="Train SC-VAE models")
parser.add_argument("--job_id", type=int, dest="job_id")
parser.add_argument("--project", "-p", type=str, dest="project")
parser.add_argument("--name", "-n", type=str, dest="name")
args = parser.parse_args()

### Set/Load Parameters
wandb.login()

if args.job_id is not None:
    z_path = Path(RESULTS_PATH + args.project)
    folders = [str(f.parts[-1]) for f in z_path.iterdir() if f.is_dir()]
    name = folders[args.job_id]
    # analysis_key = "{}/{}/".format(args.project, name)
else:
    name = args.name

# Read in config file with all parameters and settings
config = read.config(
    "{}/{}/{}/model_config.yaml".format(RESULTS_PATH, args.project, name)
)

run = wandb.init(
    project=args.project, name=name, config=config, dir=RESULTS_PATH + args.project + "/" + name
)
print("WANDB directory: {}".format(run.dir))

# Get DataLoaders and model
if "immunostain" in config["data"]["data_path"]:
    train_loader, model = scrubbed_cvae.get.data_and_model(
        config,
        dataset_label="Train",
        data_keys=["x6d", "root", "offsets", "target_pose"]
        + config["disentangle"]["features"],
        shuffle=True,
    )
    test_loader = None
elif "ensemble_healthy" in config["data"]["data_path"]:
    train_loader, test_loader, model = scrubbed_cvae.get.data_and_model(
        config,
        dataset_label="Both",
        data_keys=["x6d", "root", "offsets", "target_pose"]
        + config["disentangle"]["features"],
        shuffle=True,
    )

# Train model
model = scrubbed_cvae.train.train(config, model, train_loader, test_loader, run)

run.finish()
