import ssumo
import torch

torch.autograd.set_detect_anomaly(True)
import torch.optim as optim
import tqdm
from ssumo.params import read
import pickle
import sys
from base_path import RESULTS_PATH


def monte_carlo_mutual_information(model, data_s, z, v, gamma=0.2, device="cuda"):
    training = model.training
    model.eval()
    data_os = model.encode(data_s)
    mu_s, L_s = [d.detach() for d in data_os]
    var_s = torch.matmul(L_s, torch.transpose(L_s, dim0=-2, dim1=-1))
    exponent_s = get_gaussian_exponents(z, mu_s, var_s)

    v_s = torch.cat([data_s[k] for k in model.disentangle_keys], dim=-1)
    var_g = torch.eye(v_s.shape[-1], device=device) * gamma
    exponent_g = get_gaussian_exponents(v, v_s, var_g)
    import pdb; pdb.set_trace()
    if training:
        model.train()
    return 0


def get_gaussian_exponents(x, mu, var):
    k = x.shape[-1]
    log2pi = torch.log(torch.tensor([2 * torch.pi], device=mu.device))
    logA = -0.5*(k * log2pi + torch.sum(torch.log(var.diagonal(dim1=-2, dim2=-1)), dim=-1))

    resid = x[:, None, :] - mu[None, ...]
    inv_var = var.diagonal_scatter(1 / var.diagonal(dim1=-2, dim2=-1), dim1=-2, dim2=-1)
    resid = -0.5*(torch.bmm(resid, inv_var) * resid).sum(dim=-1)
    return logA[None,:] + resid


### Set/Load Parameters
analysis_key = sys.argv[1]

if len(sys.argv) > 2:
    job_id = sys.argv[2]
    print(job_id)
    print(sys.argv)
    analysis_key = "{}/{}/".format(analysis_key, job_id)

config = read.config(RESULTS_PATH + analysis_key + "/model_config.yaml")

### Load Dataset
dataset, loader = ssumo.data.get_mouse(
    data_config=config["data"],
    window=config["model"]["window"],
    train=True,
    data_keys=["x6d", "root", "offsets", "target_pose"]
    + config["disentangle"]["features"],
    shuffle=True,
    normalize=config["disentangle"]["features"],
)


model, device = ssumo.model.get(
    model_config=config["model"],
    disentangle_config=config["disentangle"],
    n_keypts=dataset.n_keypts,
    direction_process=config["data"]["direction_process"],
    arena_size=dataset.arena_size,
    kinematic_tree=dataset.kinematic_tree,
    verbose=1,
)

if device == "cuda":
    torch.backends.cudnn.benchmark = True

# latents = ssumo.eval.get.latents(model, dataset, config, device, "Train")

for batch_idx, data in enumerate(loader):
    data = {k: v.to(device) for k, v in data.items()}
    data["kinematic_tree"] = model.kinematic_tree
    data_o = ssumo.train.trainer.predict_batch(model, data, model.disentangle_keys)
    v = torch.cat([data[k] for k in model.disentangle_keys], dim=-1)

    monte_carlo_mutual_information(model, data, data_o["mu"], v)
