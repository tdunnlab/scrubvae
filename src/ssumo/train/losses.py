import torch.nn.functional as F
import torch
from ssumo.data.rotation_conversion import rotation_6d_to_matrix
from ssumo.data.dataset import fwd_kin_cont6d_torch


def rotation_loss(x, x_hat, eps=1e-7):
    assert x.shape[-1] == 6
    assert x_hat.shape[-1] == 6
    m1 = rotation_6d_to_matrix(x).view((-1, 3, 3))
    m2 = rotation_6d_to_matrix(x_hat).view((-1, 3, 3))

    m = torch.bmm(m1, m2.permute(0, 2, 1))  # batch*3*3

    cos = (m[:, 0, 0] + m[:, 1, 1] + m[:, 2, 2] - 1) / 2
    cos = torch.clamp(cos, -1 + eps, 1 - eps)
    theta = torch.acos(cos).mean()

    return theta


def regularize_loss(mu, log_var):
    KL_div = -0.5 * torch.sum(1 + log_var - mu.pow(2) - log_var.exp())
    return KL_div


def prior_loss(mu, L):
    var = torch.matmul(L, torch.transpose(L, dim0=-2, dim1=-1))
    KL_div = -0.5 * torch.sum(
        1
        + 2 * torch.log(L.diagonal(dim1=-1, dim2=-2))
        - mu.pow(2)
        - var.diagonal(dim1=-1, dim2=-2)
    )/mu.shape[0]
    return KL_div


def vae_BXEntropy_loss(x, x_hat, mu, log_var):
    B_XEntropy = F.binary_cross_entropy(x_hat, x.view(-1, 784), reduction="mean")
    KL_div = -0.5 * torch.sum(1 + log_var - mu.pow(2) - log_var.exp())
    return B_XEntropy + KL_div


def mpjpe_loss(pose, x_hat, kinematic_tree, offsets, root=None, root_hat=None):
    # if root == None:
    #     root = torch.zeros((x.shape[0], 3), device=x.device)
    if root_hat == None:
        root_hat = torch.zeros((x_hat.shape[0], 3), device=x_hat.device)

    # pose = fwd_kin_cont6d_torch(
    #     x, kinematic_tree, offsets, root_pos=root, do_root_R=True, eps=1e-8
    # )
    # pose = x
    pose_hat = fwd_kin_cont6d_torch(
        x_hat, kinematic_tree, offsets, root_pos=root_hat, do_root_R=True, eps=1e-8
    )

    loss = torch.mean((pose - pose_hat) ** 2)
    # loss = loss / (pose.shape[-1] * pose.shape[-2])
    return loss


def hierarchical_orthogonal_loss(L1, L2):
    Sig1 = torch.matmul(L1, torch.transpose(L1, dim0=-2, dim1=-1))
    Sig2 = torch.matmul(torch.transpose(L2, dim0=-2, dim1=-1), L2)
    return torch.sum(torch.matmul(Sig1, Sig2).diagonal(dim1=-1, dim2=-2))


def get_batch_loss(data, data_o, loss_scale):
    batch_loss = {}

    if "rotation" in loss_scale.keys():
        batch_loss["rotation"] = rotation_loss(data["x6d"], data_o["x6d"])

    if "prior" in loss_scale.keys():
        if type(data_o["mu"]) is tuple:
            batch_loss["prior"] = 0
            for mu, L in zip(data_o["mu"], data_o["L"]):
                batch_loss["prior"] += prior_loss(mu, L)
        else:
            batch_loss["prior"] = prior_loss(data_o["mu"], data_o["L"])

    if "jpe" in loss_scale.keys():
        batch_loss["jpe"] = mpjpe_loss(
            data["target_pose"].reshape(
                -1, data["x6d"].shape[-2], 3
            ),  # data["x6d"].reshape(-1, *data["x6d"].shape[-2:]),
            data_o["x6d"].reshape(-1, *data["x6d"].shape[-2:]),
            data["kinematic_tree"],
            data["offsets"].view(-1, *data["offsets"].shape[-2:]),
        )

    if "root" in loss_scale.keys():
        batch_loss["root"] = torch.nn.MSELoss(reduction="mean")(
            data_o["root"], data["root"]
        )

    available_dis_keys = ["avg_speed", "frame_speed", "part_speed", "heading_change", "heading", "avg_speed_3d"]
    num_keys = len(data_o["disentangle"].keys())
    for key in available_dis_keys:
        if key in loss_scale.keys():
            batch_loss[key] = (
                torch.nn.MSELoss(reduction="mean")(
                    data_o["disentangle"][key]["v"], data[key]
                )
                / num_keys
            )
            
        if key + "_gr" in loss_scale.keys():
            if isinstance(data_o["disentangle"][key]["gr"], list):
                batch_loss[key + "_gr"] = 0
                for gr_e in data_o["disentangle"][key]["gr"]:
                    batch_loss[key + "_gr"] += torch.nn.MSELoss(reduction="mean")( gr_e, data[key] )
                batch_loss[key + "_gr"] = (
                    batch_loss[key + "_gr"]
                    / len(data_o["disentangle"][key]["gr"])
                    / num_keys
                )
            elif torch.is_tensor(data_o["disentangle"][key]["gr"]):
                batch_loss[key + "_gr"] = (
                    torch.nn.MSELoss(reduction="mean")(
                        data_o["disentangle"][key], data[key]
                    )
                    / num_keys
                )

    # if "speed_regularize" in loss_scale.keys():
    #     batch_loss["speed_regularize"] = torch.sum(
    #         torch.diff(data_o["speed_decoder_weight"], n=2, dim=0) ** 2
    #     )

    if "orthogonal_cov" in loss_scale.keys():
        batch_loss["orthogonal_cov"] = hierarchical_orthogonal_loss(*data_o["L"])

    # for loss in batch_loss.keys():
    #     if not (loss_scale[loss] > 0):
    #         batch_loss[loss].detach()

    batch_loss["total"] = sum(
        [loss_scale[k] * batch_loss[k] for k in batch_loss.keys() if loss_scale[k] > 0]
    )

    return batch_loss
