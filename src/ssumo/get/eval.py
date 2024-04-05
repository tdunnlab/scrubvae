from pathlib import Path
from torch.utils.data import DataLoader
import tqdm
import torch
import numpy as np

def dataset_latents(model, dataset, config, device, dataset_label):
    if model is not None:
        model.eval()
    latent_path = "{}/latents/{}_{}.npy".format(
        config["out_path"], dataset_label, config["model"]["start_epoch"]
    )
    if not Path(latent_path).exists():
        loader = DataLoader(
            dataset=dataset, batch_size=config["data"]["batch_size"], shuffle=False, num_workers=5
        )
        print("Latent projections not found - Embedding dataset ...")
        latents = []
        with torch.no_grad():
            for _, data in enumerate(tqdm.tqdm(loader)):
                data = {k:v.to(device) for k,v in data.items() if k in ["x6d", "root"]}
                latents += [model.encode(data)[0].detach().cpu()]

        latents = torch.cat(latents, axis=0)
        np.save(
            latent_path,
            np.array(latents),
        )
    else:
        print("Found existing latent projections - Loading ...")
        latents = np.load(latent_path)
        assert latents.shape[0] == len(dataset)
        latents = torch.tensor(latents)

    nonzero_std_z = torch.where(latents.std(dim=0) > 0.1)[0]

    print(
        "Latent dimensions with variance over the dataset > 0.1 : {}".format(
            len(nonzero_std_z)
        )
    )
    print(latents.std(dim=0))
    return latents
