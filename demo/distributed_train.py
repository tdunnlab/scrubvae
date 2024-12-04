import os
import functools
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torchvision import datasets, transforms

import torch.distributed as dist
import torch.multiprocessing as mp
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data.distributed import DistributedSampler
from torch.distributed.fsdp import FullyShardedDataParallel as FSDP
from torch.distributed.fsdp.fully_sharded_data_parallel import (
    CPUOffload,
    BackwardPrefetch,
)
from torch.distributed.fsdp.wrap import (
    size_based_auto_wrap_policy,
    enable_wrap,
    wrap,
)

import ssumo
from ssumo.params import read
import neuroposelib
import sys
from pathlib import Path
import wandb
import argparse
from ssumo.train.trainer import predict_batch, get_optimizer_and_lr_scheduler
from ssumo.train.losses import get_batch_loss
from ssumo.get.data import calculate_mouse_kinematics
from ssumo.data.dataset import MouseDataset


RESULTS_PATH = "./"


def setup(rank, world_size):
    os.environ["MASTER_ADDR"] = "localhost"
    os.environ["MASTER_PORT"] = "12355"

    # initialize the process group
    dist.init_process_group("nccl", rank=rank, world_size=world_size)


def main(rank, world_size, config):
    setup(rank, world_size)
    skeleton_config = neuroposelib.io.read.config(config["data"]["skeleton_path"])

    datasets, samplers, loaders = {}, {}, {}
    for key in ["Train", "Test"]:
        data, window_inds = calculate_mouse_kinematics(
            config["data"],
            skeleton_config,
            config["model"]["window"],
            key == "Train",
            ["x6d", "root", "offsets", "target_pose"],
        )

        datasets[key] = MouseDataset(
            data,
            window_inds,
            config["data"]["arena_size"],
            skeleton_config["KINEMATIC_TREE"],
            len(skeleton_config["LABELS"]),
            label=key,
            discrete_classes={},
            norm_params={},
        )
        samplers[key] = DistributedSampler(
            datasets[key], rank=rank, num_replicas=world_size, shuffle=key == "Train"
        )
        loaders[key] = torch.utils.data.DataLoader(
            datasets[key],
            sampler=samplers[key],
            batch_size=config["data"]["batch_size"],
            shuffle=False,
            num_workers=5,
            pin_memory=True,
        )

    my_auto_wrap_policy = functools.partial(
        size_based_auto_wrap_policy, min_num_params=20000
    )
    torch.cuda.set_device(rank)

    init_start_event = torch.cuda.Event(enable_timing=True)
    init_end_event = torch.cuda.Event(enable_timing=True)

    model = ssumo.get.model(
        model_config=config["model"],
        load_model=None,
        epoch=0,
        disentangle_config=config["disentangle"],
        n_keypts=datasets["Train"].n_keypts,
        direction_process=config["data"]["direction_process"],
        loss_config=config["loss"],
        arena_size=datasets["Train"].arena_size,
        kinematic_tree=datasets["Train"].kinematic_tree,
        bound=None,
        discrete_classes={},
        device=rank,
        verbose=-1,
    )
    model = FSDP(model, auto_wrap_policy=my_auto_wrap_policy)

    optimizer = get_optimizer_and_lr_scheduler(
        model,
        config["train"],
        config["model"]["load_model"],
        config["model"]["start_epoch"],
    )[0]

    init_start_event.record()
    for epoch in range(1, config["train"]["num_epochs"] + 1):
        run_epoch = functools.partial(
            train_test_epoch,
            config=config,
            model=model,
            epoch=epoch,
            rank=rank,
            world_size=world_size,
        )
        run_epoch(
            loader=loaders["Train"],
            optimizer=optimizer,
            mode="train",
            sampler=samplers["Train"],
        )
        if epoch % 5 == 0:
            run_epoch(
                loader=loaders["Test"],
                mode="test",
            )

    init_end_event.record()

    if rank == 0:
        print(
            f"CUDA event elapsed time: {init_start_event.elapsed_time(init_end_event) / 1000}sec"
        )
        print(f"{model}")

    # if args.save_model:
    #     # use a barrier to make sure training is done on all ranks
    #     dist.barrier()
    #     states = model.state_dict()
    #     if rank == 0:
    #         torch.save(states, "mnist_cnn.pt")

    dist.destroy_process_group()


def train_test_epoch(
    config,
    model,
    loader,
    epoch,
    rank,
    world_size,
    optimizer=None,
    mode="train",
    sampler=None,
):
    if mode == "train":
        model.train()
        grad_env = torch.enable_grad
        if sampler:
            sampler.set_epoch(epoch)
    elif mode == "test":
        model.eval()
        grad_env = torch.no_grad
    else:
        raise ValueError("This mode is not recognized.")

    with grad_env():
        epoch_metrics = {k: torch.zeros(1).to(rank) for k in ["total"] + list(config["loss"].keys())}
        for batch_idx, data in enumerate(loader):
            data = {k: v.to(rank) for k, v in data.items()}
            data_o = predict_batch(model, data, model.disentangle_keys)
            batch_loss = get_batch_loss(
                model,
                data,
                data_o,
                config["loss"],
                config["disentangle"],
            )

            if mode == "train":
                for param in model.parameters():
                    param.grad = None

                batch_loss["total"].backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1e6)
                optimizer.step()

            epoch_metrics = {
                k: v + batch_loss[k].detach() for k, v in epoch_metrics.items()
            }

        for k, v in epoch_metrics.items():
            dist.all_reduce(v, op=dist.ReduceOp.SUM)
            if rank == 0:
                metric = v.item() / len(loader) / world_size
                print(
                    "====> {} Epoch: {} Average {} loss: {:.4f}".format(
                        mode.title(), epoch, k, metric
                    )
                )

    return epoch_metrics


if __name__ == "__main__":
    ### Set/Load Parameters
    parser = argparse.ArgumentParser(
        prog="SC-VAE Train", description="Train SC-VAE models"
    )
    parser.add_argument("--project", "-p", type=str, dest="project")
    parser.add_argument("--name", "-n", type=str, dest="name")
    args = parser.parse_args()
    name = args.name

    config = read.config(
        "{}/{}/{}/model_config.yaml".format(RESULTS_PATH, args.project, name)
    )

    WORLD_SIZE = torch.cuda.device_count()
    print(WORLD_SIZE)
    mp.spawn(main, args=(WORLD_SIZE, config), nprocs=WORLD_SIZE, join=True)
