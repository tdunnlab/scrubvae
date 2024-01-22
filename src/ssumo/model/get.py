import torch
from pathlib import Path

def get(model_config, disentangle_config, n_keypts, direction_process, arena_size=None, kinematic_tree=None, verbose=1):
    feat_dim_dict = {
        "avg_speed": 1,
        "part_speed": 4,
        "frame_speed": model_config["window"] - 1,
        "heading": 2,
        "heading_change": 1,
    }

    in_channels = n_keypts * 6
    if direction_process in ["x360","midfwd",None]:
        in_channels += 3

    invariant_dim = 0
    disentangle = None
    if disentangle_config["method"] == "invariant":
        invariant_dim = sum([feat_dim_dict[k] for k in disentangle_config["features"]])
    elif ("gr_" or "linear") in disentangle_config["method"]:
        from ssumo.model.LinearDisentangle import LinearDisentangle

        if disentangle_config["method"] == "linear":
            reversal = None
        else:
            reversal = disentangle_config["method"][3:]
        disentangle = {}
        for feat in disentangle_config["features"]:
            disentangle[feat] = LinearDisentangle(
                model_config["z_dim"],
                feat_dim_dict[feat],
                bias=False,
                reversal=reversal,
                alpha=disentangle_config["alpha"],
                do_detach=disentangle_config["detach_gr"],
            )
    
    ### Initialize/load model
    if model_config["type"] == "rcnn":
        from ssumo.model.ResVAE import ResVAE
        vae = ResVAE(
            in_channels=in_channels,
            kernel=model_config["kernel"],
            z_dim=model_config["z_dim"],
            window=model_config["window"],
            activation=model_config["activation"],
            is_diag=model_config["diag"],
            invariant_dim=invariant_dim,
            init_dilation=model_config["init_dilation"],
            disentangle=disentangle,
            disentangle_keys=disentangle_config["features"],
            arena_size = arena_size,
            kinematic_tree = kinematic_tree,
        )
    elif model_config["type"] == "transformer":
        from ssumo.model.TransformerVAE import TransformerVAE

        vae = TransformerVAE(
            in_channels=in_channels,
            window=model_config["window"],
            z_dim=model_config["z_dim"],
            activation=model_config["activation"],
            is_diag=model_config["diag"],
            n_heads=model_config["n_heads"],
        )
    elif model_config["type"] == "hrcnn":
        from ssumo.model.ResVAE import HResVAE

        vae = HResVAE(
            in_channels=in_channels,
            kernel=model_config["kernel"],
            z_dim=model_config["z_dim"],
            window=model_config["window"],
            activation=model_config["activation"],
            is_diag=model_config["diag"],
            invariant_dim=invariant_dim,
            init_dilation=model_config["init_dilation"],
        )
    if verbose > 0:
        print(vae)

    if model_config["load_model"] is not None:
        load_path = "{}/weights/epoch_{}.pth".format(
            model_config["load_model"], model_config["start_epoch"]
        )
        print("Loading Weights from:\n{}".format(load_path))
        # import pdb; pdb.set_trace()
        state_dict = torch.load(load_path)
        state_dict["arena_size"] = arena_size.cuda()
        vae.load_state_dict(state_dict)

        # spd_decoder_path = "{}/weights/{}_spd_epoch_{}.pth".format(
        #     model_config["load_model"],
        #     model_config["speed_decoder"],
        #     model_config["load_epoch"],
        # )
        # if Path(spd_decoder_path).exists() and (
        #     model_config["speed_decoder"] is not None
        # ):
        #     print(
        #         "Found {} speed decoder weights - Loading ...".format(
        #             model_config["speed_decoder"]
        #         )
        #     )
        #     speed_decoder.load_state_dict(torch.load(spd_decoder_path))

    device = "cuda:0" if torch.cuda.is_available() else "cpu"

    # if model_config["speed_decoder"] == None:
    return vae.to(device), device
    # else:
    #     return vae.to(device), speed_decoder.to(device) device
