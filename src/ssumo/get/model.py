import torch

def model(
    model_config,
    load_model,
    epoch,
    disentangle_config,
    n_keypts,
    direction_process,
    arena_size=None,
    kinematic_tree=None,
    bound=False,
    device="cuda",
    verbose=1,
):
    feat_dim_dict = {
        "avg_speed": 1,
        "part_speed": 4,
        "frame_speed": model_config["window"] - 1,
        "avg_speed_3d": 3,
        "heading": 2,
        "heading_change": 1,
    }

    in_channels = n_keypts * 6
    if direction_process in ["x360", "midfwd", None]:
        in_channels += 3

    conditional_dim = 0
    disentangle = None
    if disentangle_config["method"]:
        if "conditional" in disentangle_config["method"]:
            conditional_dim = sum(
                [feat_dim_dict[k] for k in disentangle_config["features"]]
            )

    if disentangle_config["method"] is None:
        pass
    elif disentangle_config["method"] == "gr_conditional":
        from ssumo.model.disentangle import Scrubber

        disentangle = {}
        for feat in disentangle_config["features"]:
            disentangle[feat] = Scrubber(
                model_config["z_dim"],
                feat_dim_dict[feat],
                disentangle_config["alpha"],
                bound=bound,
            )
    elif ("gr_" in disentangle_config["method"]) or (
        "linear" in disentangle_config["method"]
    ):
        from ssumo.model.disentangle import LinearDisentangle

        if disentangle_config["method"] == "linear":
            reversal = None
        elif disentangle_config["method"] == "gr_conditional":
            reversal = "conditional"
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
                n_models=disentangle_config["n_models"],
            )
            
    elif disentangle_config["moving_avg_lsq"] is True:
        from ssumo.model.disentangle import MovingAvgLeastSquares
        disentangle = {}
        for feat in disentangle_config["features"]:
            disentangle[feat] = MovingAvgLeastSquares(
                model_config["z_dim"], feat_dim_dict[feat]
            )

    ### Initialize/load model
    if model_config["type"] == "rcnn":
        from ssumo.model.residual import ResVAE

        vae = ResVAE(
            in_channels=in_channels,
            kernel=model_config["kernel"],
            z_dim=model_config["z_dim"],
            window=model_config["window"],
            activation=model_config["activation"],
            is_diag=model_config["diag"],
            conditional_dim=conditional_dim,
            init_dilation=model_config["init_dilation"],
            disentangle=disentangle,
            disentangle_keys=disentangle_config["features"],
            arena_size=arena_size,
            kinematic_tree=kinematic_tree,
            ch=model_config["channel"],
        )

    if verbose > 0:
        print(vae)

    if load_model is not None:
        load_path = "{}/weights/epoch_{}.pth".format(
            load_model, epoch
        )
        print("Loading Weights from:\n{}".format(load_path))
        state_dict = torch.load(load_path)
        missing_keys, unexpected_keys = vae.load_state_dict(state_dict, strict=False)

        if verbose > 0:
            print("Missing Keys: {}".format(missing_keys))
            print("Unexpected Keys: {}".format(unexpected_keys))

    return vae.to(device)
