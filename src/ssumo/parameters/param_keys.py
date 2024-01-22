PARAM_KEYS = dict(
    data=[
        "arena_size",
        "data_path",
        "direction_process",
        "filter_pose",
        "remove_speed_outliers",
        "skeleton_path",
        "stride",
    ],
    disentangle=["alpha", "detach_gr", "features", "method"],
    model=[
        "activation",
        "diag",
        "init_dilation",
        "kernel",
        "start_epoch",
        "load_model",
        "type",
        "window",
        "z_dim",
    ],
    train=["beta_anneal", "batch_size", "num_epochs"],
)
