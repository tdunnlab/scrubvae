import yaml
from dappy import read
from pathlib import Path
from ssumo.params import PARAM_KEYS
from pathlib import Path

def config(path):
    config = read.config(path)

    for key in PARAM_KEYS.keys():
        if key not in config.keys():
            config[key] = {}

        for param in PARAM_KEYS[key]:
            if param not in config[key].keys():
                config[key][param] = None

    if config["disentangle"]["features"] == None:
        config["disentangle"]["features"] = []

    if config["out_path"] == "current":
        config["out_path"] = str(Path(path).parent) + "/"
    
    print("Saving folder: {}".format(config["out_path"]))

    sub_folders = ["weights/", "losses/", "latents/"]
    for folder in sub_folders:
        Path(config["out_path"] + folder).mkdir(parents=True, exist_ok=True)

    f = open(config["out_path"] + "/model_config.yaml", "w")
    yaml.dump(config, f)
    f.close()

    return config

# def command_arg(path):




