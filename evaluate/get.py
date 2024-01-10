from pathlib import Path
from torch.utils.data import DataLoader
import tqdm
import torch
import numpy as np

def latents(model, dataset, config, device, dataset_label):
    model.eval()
    latent_path = "{}/latents/{}_{}.npy".format(
        config["out_path"], dataset_label, config["load_epoch"]
    )
    if not Path(latent_path).exists():
        loader = DataLoader(
            dataset=dataset, batch_size=config["batch_size"], shuffle=False
        )
        print("Latent projections not found - Embedding dataset ...")
        latents = []
        with torch.no_grad():
            for _, data in enumerate(tqdm.tqdm(loader)):
                x6d = data["x6d"].to(device)

                if config["arena_size"] is not None:
                    root = data["root"].to(device)
                    x_i = torch.cat((x6d.view(x6d.shape[:2] + (-1,)), root), axis=-1)
                    mu, L = model.encoder(x_i.moveaxis(1, -1))
                else:
                    mu, L = model.encoder(x6d)

                latents += [mu.detach().cpu()]

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