from torch.autograd import Function
import torch.nn as nn
import torch
from torch.nn.functional import mse_loss


class MovingAvgLeastSquares(torch.nn.Module):

    def __init__(self, nx, ny, lamdiff=1e-1, delta=1e-4):
        super().__init__()
        # Running average of covariances for first linear decoder
        self.register_buffer("Sxx0", torch.eye(nx))
        self.register_buffer("Sxy0", torch.zeros(nx, ny))

        # Running average of covariances for first linear decoder
        self.register_buffer("Sxx1", torch.eye(nx))
        self.register_buffer("Sxy1", torch.zeros(nx, ny))

        # Forgetting factor for the first linear decoder
        self.lam0 = 0.83

        # Forgetting factor for the second linear decoder
        self.lam1 = self.lam0 + lamdiff

        # Update parameters for the forgetting factors
        self.delta = delta
        self.lamdiff = lamdiff

    def forward(self, x):
        # Solve optimal decoder weights (normal equations)
        W0 = torch.linalg.solve(self.Sxx0, self.Sxy0)
        W1 = torch.linalg.solve(self.Sxx1, self.Sxy1)

        # Predicted values for y
        yhat0 = x @ W0
        yhat1 = x @ W1
        return [yhat0, yhat1]

    def evaluate_loss(self, yhat0, yhat1, x, y):
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
        # Loss for each decoder
        l0 = mse_loss(y, yhat0)
        l1 = mse_loss(y, yhat1)

        # If lam0 performed better than lam1, we decrease the forgetting factors
        # by self.delta
        if l0 < l1:
            self.lam0 = torch.clamp(self.lam0 - self.delta, 0.0, 1.0)
            self.lam1 = self.lam0 + self.lamdiff

        # If lam1 performed better than lam0, we increase the forgetting factors
        # by self.delta
        else:
            self.lam1 = torch.clamp(self.lam1 + self.delta, 0.0, 1.0)
            self.lam0 = self.lam1 - self.lamdiff

        # Compute moving averages for the next batch of data
        self.Sxx0 = self.lam0 * self.Sxx0 + x.T @ x
        self.Sxy0 = self.lam0 * self.Sxy0 + x.T @ y
        self.Sxx1 = self.lam1 * self.Sxx1 + x.T @ x
        self.Sxy1 = self.lam1 * self.Sxy1 + x.T @ y

        print("Lam0: {:4f},   Lam1: {:4f}".format(self.lam0, self.lam1))

        # Return average loss of the two linear decoders
        return (l0 + l1) * 0.5


class GradientReversal(Function):
    @staticmethod
    def forward(ctx, x, alpha):
        ctx.save_for_backward(x, alpha)
        return x

    @staticmethod
    def backward(ctx, grad_output):
        grad_input = None
        _, alpha = ctx.saved_tensors
        if ctx.needs_input_grad[0]:
            grad_input = -alpha * grad_output
        return grad_input, None


revgrad = GradientReversal.apply


class GradientReversalLayer(nn.Module):
    def __init__(self, alpha):
        super(GradientReversalLayer, self).__init__()
        self.alpha = torch.tensor(alpha, requires_grad=False)

    def forward(self, x):
        return revgrad(x, self.alpha)


class MLP(nn.Module):
    def __init__(self, in_dim, out_dim):
        super(MLP, self).__init__()
        self.mlp = nn.Sequential(
            nn.Linear(in_dim, in_dim),
            nn.ReLU(),
            nn.Linear(in_dim, in_dim),
            nn.ReLU(),
            nn.Linear(in_dim, out_dim),
        )

    def forward(self, z):
        return self.mlp(z)


class MLPEnsemble(nn.Module):
    def __init__(self, in_dim, out_dim, n_models=3):
        super(MLPEnsemble, self).__init__()
        mlp_list = []
        for i in range(n_models):
            mlp_list += [MLP(in_dim, out_dim)]
        self.ensemble = nn.ModuleList(mlp_list)

    def forward(self, z):
        return [mlp(z) for mlp in self.ensemble]


class ReversalEnsemble(nn.Module):
    def __init__(self, in_dim, out_dim, bound=False):
        super(ReversalEnsemble, self).__init__()

        # self.lin = nn.Sequential(
        #     nn.Linear(in_dim, out_dim),
        #     nn.Tanh() if bound else None,
        # )

        self.mlp1 = nn.Sequential(
            nn.Linear(in_dim, in_dim),
            nn.ReLU(),
            nn.Linear(in_dim, in_dim),
            nn.ReLU(),
            nn.Linear(in_dim, out_dim),
            nn.Tanh() if bound else None,
        )

        self.mlp2 = nn.Sequential(
            nn.Linear(in_dim, in_dim),
            nn.ReLU(),
            nn.Linear(in_dim, out_dim),
            nn.Tanh() if bound else None,
        )

        self.mlp3 = nn.Sequential(
            nn.Linear(in_dim, in_dim),
            nn.ReLU(),
            nn.Linear(in_dim, in_dim // 2),
            nn.ReLU(),
            nn.Linear(in_dim // 2, out_dim),
            nn.Tanh() if bound else None,
        )

    def forward(self, z):
        # a = self.lin(z)
        b = self.mlp1(z)
        c = self.mlp2(z)
        d = self.mlp3(z)
        return [b, c, d]  # a,


class Scrubber(nn.Module):
    def __init__(self, in_dim, out_dim, alpha=1.0, bound=False):
        super(Scrubber, self).__init__()
        self.reversal = nn.Sequential(
            GradientReversalLayer(alpha), ReversalEnsemble(in_dim, out_dim, bound)
        )

    def forward(self, z):
        return {"gr": self.reversal(z)}


class LinearDisentangle(nn.Module):
    def __init__(
        self,
        in_dim,
        out_dim,
        bias=False,
        reversal="linear",
        alpha=1.0,
        do_detach=True,
        n_models=None,
    ):
        super(LinearDisentangle, self).__init__()
        self.do_detach = do_detach

        self.decoder = nn.Linear(in_dim, out_dim, bias=bias)
        if reversal == "mlp":
            self.reversal = nn.Sequential(
                GradientReversalLayer(alpha),
                nn.Linear(in_dim, in_dim),
                nn.ReLU(),
                nn.Linear(in_dim, in_dim),
                nn.ReLU(),
                nn.Linear(in_dim, out_dim),
            )
        elif reversal == "linear":
            self.reversal = nn.Sequential(
                GradientReversalLayer(alpha), nn.Linear(in_dim, out_dim, bias=True)
            )
        elif reversal == "ensemble":
            if (n_models == None) or (n_models == 0):
                self.reversal = nn.Sequential(
                    GradientReversalLayer(alpha), ReversalEnsemble(in_dim, out_dim)
                )
            else:
                self.reversal = nn.Sequential(
                    GradientReversalLayer(alpha), MLPEnsemble(in_dim, out_dim, n_models)
                )
        else:
            self.reversal = None

    def forward(self, z):
        x = self.decoder(z)

        if self.reversal is not None:
            w = self.decoder.weight

            nrm = w @ w.T
            z_sub = z - torch.linalg.solve(nrm, x.T).T @ w

            return {"v": x, "gr": self.reversal(z_sub)}

        return {"v": x}
