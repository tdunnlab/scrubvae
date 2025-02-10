import yaml
from neuroposelib import read
from pathlib import Path
from scrubvae.params import PARAM_KEYS
from pathlib import Path


def config(path):
    config = read.config(path)

    for key in PARAM_KEYS.keys():
        if key not in config.keys():
            config[key] = {}

        for param in PARAM_KEYS[key]:
            if param not in config[key].keys():
                config[key][param] = None

    if (not config["disentangle"]["method"]) or (config["disentangle"]["method"] is None):
        config["disentangle"]["method"] = {}
    
    if (config["disentangle"]["features"] is None) or (len(config["disentangle"]["features"]) < 1):
        all_feats = []
        for k, v in config["disentangle"]["method"].items():
            all_feats += v
        all_feats = list(set(all_feats))
        config["disentangle"]["features"] = all_feats

    if config["out_path"] == "current":
        config["out_path"] = str(Path(path).parent) + "/"
    
    print("Saving folder: {}".format(config["out_path"]))

    sub_folders = ["weights/", "checkpoints/", "latents/"]
    for folder in sub_folders:
        Path(config["out_path"] + folder).mkdir(parents=True, exist_ok=True)

    f = open(config["out_path"] + "/model_config.yaml", "w")
    yaml.dump(config, f)
    f.close()

    return config

# def command_arg(path):




