import torch


def model(
    model_config,
    load_model,
    epoch,
    disentangle_config,
    n_keypts,
    direction_process,
    loss_config=None,
    arena_size=None,
    kinematic_tree=None,
    bound=False,
    discrete_classes=None,
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
        "fluorescence": 1,
    }

    if discrete_classes is not None:
        print("Discrete Classes: {}".format(discrete_classes))
        feat_dim_dict.update({k: len(v) for k, v in discrete_classes.items()})

    in_channels = n_keypts * 6
    if direction_process in ["x360", "midfwd", None]:
        in_channels += 3

    methods = disentangle_config["method"]
    disentangle = {}
    # Linear Projection model for each disentanglement feature
    if "linear" in methods.keys():  # len(methods["linear"]) > 0:
        from scrubbed_cvae.model.disentangle import LinearProjection

        disentangle["linear"] = {}
        for feat in methods["linear"]:
            disentangle["linear"][feat] = LinearProjection(
                model_config["z_dim"],
                feat_dim_dict[feat],
                bias=False,
            )

    # Conditional VAE adding dimensions to decoder input
    if "conditional" in methods.keys():  # len(methods["conditional"]) > 0:
        conditional_dim = sum([feat_dim_dict[k] for k in methods["conditional"]])
        conditional_keys = methods["conditional"]
    else:
        conditional_keys = None
        conditional_dim = 0

    # Gradient reversal scrubbing for each disentanglement feature
    if "grad_reversal" in methods.keys():  # len(methods["grad_reversal"]) > 0:
        from scrubbed_cvae.model.disentangle import GRScrubber

        disentangle["grad_reversal"] = {}
        for feat in methods["grad_reversal"]:
            disentangle["grad_reversal"][feat] = GRScrubber(
                model_config["z_dim"],
                feat_dim_dict[feat],
                alpha=disentangle_config["alpha"],
                bound=bound,
            )

    # Moving Average Least Squares Filter with n-order polynomical features
    if "moving_avg_lsq" in methods.keys():  # len(methods["moving_avg_lsq"]) > 0:
        from scrubbed_cvae.model.disentangle import MovingAvgLeastSquares

        disentangle["moving_avg_lsq"] = {}
        for feat in methods["moving_avg_lsq"]:
            disentangle["moving_avg_lsq"][feat] = MovingAvgLeastSquares(
                model_config["z_dim"],
                feat_dim_dict[feat],
                bias=loss_config[feat + "_mals"] < 0,
                polynomial_order=disentangle_config["polynomial"],
                l2_reg=disentangle_config["l2_reg"],
            )

    # Quadratic Discriminant Filter for class scrubbing
    if "qda" in methods.keys():  # len(methods["qda"]) > 0:
        from scrubbed_cvae.model.disentangle import QuadraticDiscriminantFilter

        disentangle["qda"] = {}
        for feat in methods["qda"]:
            disentangle["qda"][feat] = QuadraticDiscriminantFilter(
                model_config["z_dim"], discrete_classes[feat]
            )

    # Moving Average Filter for class scrubbing
    if "moving_avg" in methods.keys():  # len(methods["moving_avg"]) > 0:
        from scrubbed_cvae.model.disentangle import MovingAverageFilter

        disentangle["moving_avg"] = {}
        for feat in methods["moving_avg"]:
            disentangle["moving_avg"][feat] = MovingAverageFilter(
                model_config["z_dim"], discrete_classes[feat]
            )

    if "adversarial_net" in methods.keys():
        from scrubbed_cvae.model.disentangle import AdvNetScrubber

        disentangle["adversarial_net"] = {}
        for feat in methods["adversarial_net"]:
            disentangle["adversarial_net"][feat] = AdvNetScrubber(
                model_config["z_dim"] + conditional_dim
            )

    ### Initialize/load model
    if model_config["type"] == "rcnn":
        from scrubbed_cvae.model.residual import ResVAE

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
            conditional_keys=conditional_keys,
            arena_size=arena_size,
            kinematic_tree=kinematic_tree,
            prior=model_config["prior"],
            ch=model_config["channel"],
            discrete_classes=discrete_classes,
        )

    if verbose > 0:
        print(vae)

    if load_model is not None:
        load_path = "{}/weights/epoch_{}.pth".format(load_model, epoch)
        print("Loading Weights from:\n{}".format(load_path))
        state_dict = torch.load(load_path)
        missing_keys, unexpected_keys = vae.load_state_dict(state_dict, strict=False)

        if verbose > 0:
            print("Missing Keys: {}".format(missing_keys))
            print("Unexpected Keys: {}".format(unexpected_keys))

    return vae.to(device)
