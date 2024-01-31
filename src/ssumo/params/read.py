import yaml
from dappy import read
from pathlib import Path
from ssumo.params import PARAM_KEYS

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

    print("Saving folder: {}".format(config["out_path"]))

    sub_folders = ["weights/", "losses/", "latents/"]
    for folder in sub_folders:
        Path(config["out_path"] + folder).mkdir(parents=True, exist_ok=True)

    f = open(config["out_path"] + "/model_config.yaml", "w")
    yaml.dump(config, f)
    f.close()

    return config

# def read_config_old(path):
#     config = read.config(path)

#     decoder_type = "" if config["speed_decoder"] is None else config["speed_decoder"] + "spd"
#     decoder_type += (
#         "_"
#         if "speed" not in config["loss_scale"].keys()
#         else "{}_".format(config["loss_scale"]["speed"])
#     )

#     cov_type = "diag" if config["is_diag"] else "full"

#     norm_arena = "" if config["arena_size"] is None else "a"
#     face_dir = (
#         "{}360".format(norm_arena)
#         if config["direction_process"] is None
#         else config["direction_process"]
#     )

#     model_str = config["model_type"][:2] + "_"
#     invariant_str = (
#         "" if config["invariant"] is None else (config["invariant"][0] + "i_")
#     )
#     hier_orthgnl_scale = (
#         "_o{}".format(config["loss_scale"]["hier_orthogonal"])
#         if "hier_orthogonal" in config["loss_scale"].keys()
#         else ""
#     )

#     if "prior" in config["loss_scale"].keys():
#         beta_str = (
#             "_b{}".format(config["loss_scale"]["prior"])
#             if config["loss_scale"]["prior"] != 20
#             else ""
#         )
#     else:
#         beta_str = ""

#     beta_anneal_str = ""
#     if "beta_anneal" in config.keys():
#         if config["beta_anneal"]:
#             beta_anneal_str = "_ba"

#     grd_reversal = (
#         "gr{}{}_".format(config["gradient_reversal"][0],config["alpha"])
#         if config["gradient_reversal"]
#         else ""
#     )

#     dilation_str = (
#         ""
#         if config["init_dilation"] is None
#         else "_d{}".format(config["init_dilation"])
#     )

#     if "detach_gr" in config.keys():
#         grd_reversal = (
#             "d{}".format(grd_reversal) if config["detach_gr"] else grd_reversal
#         )

#     z_str = "" if config["z_dim"] == 128 else "_z{}".format(config["z_dim"])

#     analysis_key = "{}{}{}{}w{}{}{}_{}_{}{}{}{}".format(
#         decoder_type,
#         invariant_str,
#         grd_reversal,
#         model_str,
#         config["window"],
#         beta_str,
#         beta_anneal_str,
#         face_dir,
#         cov_type,
#         hier_orthgnl_scale,
#         z_str,
#         dilation_str
#     )
#     config["out_path"] = config["base_path"] + analysis_key + "/"
#     print("Saving folder: {}".format(config["out_path"]))

#     sub_folders = ["weights/", "losses/", "latents/"]
#     for folder in sub_folders:
#         Path(config["out_path"] + folder).mkdir(parents=True, exist_ok=True)

#     f = open(config["out_path"] + "/model_config.yaml", "w")
#     yaml.dump(config, f)
#     f.close()

#     return config



