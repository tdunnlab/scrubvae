import torch
from torch.nn.functional import mse_loss
import numpy as np
import ssumo
from dappy import read
from base_path import RESULTS_PATH
from torch.utils.data import DataLoader
from sklearn.metrics import r2_score

class MovingAvgLeastSquares:

    def __init__(self, nx, ny, lamdiff=1e-1, delta=1e-4, device="cuda"):

        # Running average of covariances for first linear decoder
        self.Sxx0 = torch.eye(nx, device=device)
        self.Sxy0 = torch.zeros(nx, ny, device=device)

        # Running average of covariances for first linear decoder
        self.Sxx1 = torch.eye(nx, device=device)
        self.Sxy1 = torch.zeros(nx, ny, device=device)

        # Forgetting factor for the first linear decoder
        self.lam0 = 0.83
        
        # Forgetting factor for the second linear decoder
        self.lam1 = self.lam0 + lamdiff

        # Update parameters for the forgetting factors
        self.delta = delta
        self.lamdiff = lamdiff
    
    def evaluate_loss(self, x, y):
        """
        Parameters
        ----------
        x : torch.tensor
            (nx x batch_size) matrix of independent variables.
        
        y : torch.tensor
            (ny x batch_size) matrix of dependent variables.
        
        Returns
        -------
        loss : torch.tensor
            Scalar loss reflecting average mean squared error of the
            two moving average estimates of the linear decoder.
        """
        
        # Solve optimal decoder weights (normal equations)
        W0 = torch.linalg.solve(self.Sxx0, self.Sxy0)
        W1 = torch.linalg.solve(self.Sxx1, self.Sxy1)

        # Predicted values for y
        yhat0 = x @ W0
        yhat1 = x @ W1

        # Loss for each decoder
        l0 = mse_loss(y, yhat0)
        l1 = mse_loss(y, yhat1)

        # If lam0 performed better than lam1, we decrease the forgetting factors
        # by self.delta
        if (l0 < l1):
            self.lam0 = np.clip(self.lam0 - self.delta, 0.0, 1.0)
            self.lam1 = self.lam0 + self.lamdiff

        # If lam1 performed better than lam0, we increase the forgetting factors
        # by self.delta
        else:
            self.lam1 = np.clip(self.lam1 + self.delta, 0.0, 1.0)
            self.lam0 = self.lam1 - self.lamdiff
        
        # Compute moving averages for the next batch of data
        self.Sxx0 = self.lam0 * self.Sxx0 + x.T @ x
        self.Sxy0 = self.lam0 * self.Sxy0 + x.T @ y
        self.Sxx1 = self.lam1 * self.Sxx1 + x.T @ x
        self.Sxy1 = self.lam1 * self.Sxy1 + x.T @ y

        print("Lam0: {:4f},   Lam1: {:4f}".format(self.lam0, self.lam1))

        # Return average loss of the two linear decoders
        return (l0 + l1) * 0.5, (yhat0 + yhat1) * 0.5
    
path = "/mcmi_32/vanilla/"
config = read.config(RESULTS_PATH + path + "/model_config.yaml")
config["model"]["load_model"] = config["out_path"]
config["model"]["start_epoch"] = 600

dataset_label = "Train"
dataset, _ = ssumo.data.get_mouse(
    data_config=config["data"],
    window=config["model"]["window"],
    train=dataset_label == "Train",
    data_keys=["heading"],
    normalize=config["disentangle"]["features"],
    shuffle=False
)

z = np.load(config["out_path"] + "latents/Train_600.npy")
dataset.data["z"] = torch.tensor(z)

loader = DataLoader(
        dataset=dataset,
        batch_size=config["data"]["batch_size"],
        shuffle=True,
        num_workers=5,
        pin_memory=True,
    )

mals = MovingAvgLeastSquares(32, 2)
for batch_idx, data in enumerate(loader):
    mse, y_hat= mals.evaluate_loss(data["z"].cuda(), data["heading"].cuda())
    print(r2_score(data["heading"], y_hat.cpu().numpy()))
