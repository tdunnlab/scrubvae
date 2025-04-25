from torch.autograd import Function
import torch.nn as nn
import torch
from torch.nn.functional import mse_loss
import numpy as np
import torch.optim as optim


class MovingAverageFilter(nn.Module):
    """
    Stores a moving average over streaming minibatches of data
    with an adaptive forgetting factor.
    """

    def __init__(self, nx, classes, lamdiff=1e-2, delta=1e-3):
        super().__init__()

        self.classes = classes
        # Running averages of means
        self.register_buffer("m1", torch.zeros(len(self.classes), nx))
        self.register_buffer("m2", torch.zeros(len(self.classes), nx))

        # Forgetting factors
        self.register_buffer("lam1", torch.ones(len(self.classes)) * 0.5)
        self.register_buffer("lam2", self.lam1 + lamdiff)

        # Update parameters for the forgetting factors
        self.delta = delta
        self.lamdiff = lamdiff

    def forward(self, *args, **kwargs):
        return

    def evaluate_loss(self, x, y):
        """
        Parameters
        ----------
        x : torch.tensor
            (batch_size x num_features) minibatch of data.

        Returns
        -------
        mean_est : torch.tensor
            (num_features,) estimate of the mean
        """
        m1 = torch.zeros_like(self.m1)
        m2 = torch.zeros_like(self.m2)
        for i, label in enumerate(self.classes):
            # Empirical mean on minibatch of data.
            xbar = torch.mean(x[(y == label).ravel(), :], axis=0)

            # See whether m1 or m2 is a closer match
            d1 = torch.linalg.norm(xbar - self.m1[i])
            d2 = torch.linalg.norm(xbar - self.m2[i])

            # Update forgetting factors
            if d1 < d2:
                self.lam1[i] = torch.clamp(self.lam1[i] - self.delta, 0.0, 1.0)
                self.lam2[i] = self.lam1[i] + self.lamdiff
            else:
                self.lam2[i] = torch.clamp(self.lam2[i] + self.delta, 0.0, 1.0)
                self.lam1[i] = self.lam2[i] - self.lamdiff

            # Update m1 and m2
            m1[i] = (1 - self.lam1[i]) * xbar + self.lam1[i] * self.m1[i]
            m2[i] = (1 - self.lam2[i]) * xbar + self.lam2[i] * self.m2[i]

        mean_estimate = 0.5 * (m1 + m2)
        d = torch.triu(
            mean_estimate.T[..., None] - mean_estimate.T[..., None, :], diagonal=1
        )

        # Return estimate of mean
        return torch.linalg.norm(d)

    def update(self, x, y):
        for i, label in enumerate(self.classes):
            # Empirical mean on minibatch of data.
            xbar = torch.mean(x[(y == label).ravel(), :], axis=0)

            # Update m1 and m2
            self.m1[i] = (1 - self.lam1[i]) * xbar + self.lam1[i] * self.m1[i]
            self.m2[i] = (1 - self.lam2[i]) * xbar + self.lam2[i] * self.m2[i]

        self.m1 = self.m1.detach()
        self.m2 = self.m2.detach()
        return self


class QuadraticDiscriminantFilter(nn.Module):
    """
    Trains a two quadratic binary classifiers with streaming minibatches of data.

    The forgetting rates of the two classifiers are automatically tuned.
    """

    def __init__(self, nx, classes, lamdiff=1e-2, delta=1e-3):
        super().__init__()

        # Running averages of means
        self.classes = classes
        n_classes = len(classes)

        param_names = [
            "0a",
            "1a",
            "0b",
            "1b",
        ]
        for name in param_names:
            self.register_buffer(
                "m{}".format(name),
                torch.zeros(n_classes, nx, requires_grad=False),
            )
            self.register_buffer(
                "S{}".format(name),
                torch.eye(nx, requires_grad=False)[None, :].repeat(n_classes, 1, 1),
            )

        self.register_buffer("lama", torch.ones(n_classes, requires_grad=False) * 0.2)
        self.register_buffer("lamb", self.lama + lamdiff)
        # Update parameters for the forgetting factors
        self.delta = delta
        self.lamdiff = lamdiff

    def forward(self, *args, **kwargs):
        return

    def cgll(self, x, m, S):
        """
        Compute Gaussian Log Likelihood
        """
        resids = torch.sum((x - m) * torch.linalg.solve(S, (x - m).T).T, axis=1)
        return -0.5 * (torch.logdet(S) + resids)

    def update(self, x, y):
        for i, label in enumerate(self.classes):
            i0 = (y != label).ravel()
            i1 = (y == label).ravel()
            # Empirical mean for -1/+1 class labels
            x0m = torch.mean(x[i0], axis=0, keepdim=True)
            x1m = torch.mean(x[i1], axis=0, keepdim=True)

            # Empirical covariance for -1/+1 class labels
            x0S = torch.cov(x[i0].T, correction=0)
            x1S = torch.cov(x[i1].T, correction=0)

            # Update classifier A/B, with forgetting factor `lama/b'
            # Update classifier A, with forgetting factor `lama`
            self.m0a[i] = (1 - self.lama[i]) * self.m0a[i] + self.lama[i] * x0m
            self.m1a[i] = (1 - self.lama[i]) * self.m1a[i] + self.lama[i] * x1m

            self.S0a[i] = (1 - self.lama[i]) * self.S0a[i] + self.lama[i] * x0S
            self.S1a[i] = (1 - self.lama[i]) * self.S1a[i] + self.lama[i] * x1S

            # Update classifier B, with forgetting factor `lamb`
            self.m0b[i] = (1 - self.lamb[i]) * self.m0b[i] + self.lamb[i] * x0m
            self.m1b[i] = (1 - self.lamb[i]) * self.m1b[i] + self.lamb[i] * x1m

            self.S0b[i] = (1 - self.lamb[i]) * self.S0b[i] + self.lamb[i] * x0S
            self.S1b[i] = (1 - self.lamb[i]) * self.S1b[i] + self.lamb[i] * x1S

        return self

    def evaluate_loss(self, x, y, update=True):
        """
        Parameters
        ----------
        x : torch.tensor
            (batch_size x nx) matrix of independent variables.

        y : torch.tensor
            (batch_size) vector of +1/-1 class labels,

        Returns
        -------
        log_likelihood : torch.tensor
            Average log likelihood of the two quadratic decoders.
        """

        ll_loss = 0
        for i, label in enumerate(self.classes):
            # Indices for -1/+1 class labels
            i0 = (y != label).ravel()
            i1 = (y == label).ravel()

            # Compute log likelihood score for classifier A
            lla0 = self.cgll(
                x,
                self.m0a[i : i + 1],
                self.S0a[i],
            )
            lla1 = self.cgll(
                x,
                self.m1a[i : i + 1],
                self.S1a[i],
            )
            lla = torch.sum(i0 * lla0 + i1 * lla1)

            # Compute log likelihood score for classifier B
            llb0 = self.cgll(
                x,
                self.m0b[i : i + 1],
                self.S0b[i],
            )
            llb1 = self.cgll(
                x,
                self.m1b[i : i + 1],
                self.S1b[i],
            )
            llb = torch.sum(i0 * llb0 + i1 * llb1)

            # If classifier A is better than B, we decrease the forgetting factors
            # by self.delta
            if update and (lla > llb):
                self.lama[i] = torch.clamp(self.lama[i] - self.delta, 0.0, 1.0)
                self.lamb[i] = self.lama[i] + self.lamdiff

            # If classifier B is better than A, we decrease the forgetting factors
            # by self.delta
            elif update:
                self.lamb[i] = torch.clamp(self.lamb[i] + self.delta, 0.0, 1.0)
                self.lama[i] = self.lamb[i] - self.lamdiff

            # Return average log-likelihood ratios of the two linear decoders
            batch_y = (i1 * 2 - 1).float()
            llra = batch_y @ (lla1 - lla0)
            llrb = batch_y @ (llb1 - llb0)

            ll_loss += (llra + llrb) * 0.5

        return ll_loss / len(self.classes)

class MutInfoEstimator(torch.nn.Module):

    def __init__(
        self, x_s, y_s, bandwidth, var_mode="sphere", model_var=None, device="cuda"
    ):
        """
        x_s: array with shape (num_s, x_dim)
        y_s: array with shape (num_s, y_dim)
        var_s: float or array/tensor (num_s, x_dim)
        gamma: float
        device: "cuda" or "cpu"
        """
        super().__init__()

        self.register_buffer("x_s", x_s)
        self.register_buffer("y_s", y_s)
        log2pi = torch.log(torch.tensor([2 * torch.pi], device=device))
        self.num_s = x_s.shape[0]
        assert y_s.shape[0] == self.num_s
        self.x_dim = x_s.shape[1]
        self.y_dim = y_s.shape[1]
        self.var_mode = var_mode

        if self.var_mode == "sphere":
            self.register_buffer(
                "var_s", torch.tensor([bandwidth], device=device, requires_grad=False)
            )
            logA_x = self.x_dim * (log2pi + torch.log(self.var_s))
        elif self.var_mode == "diagonal":
            self.register_buffer(
                "var_s", model_var.diagonal(dim1=-2, dim2=-1) ** 2 + bandwidth
            )
            logA_x = (self.x_dim * log2pi + torch.sum(torch.log(self.var_s), dim=-1))[
                None, :
            ]

        self.gamma = bandwidth

        logA_y = self.y_dim * (
            log2pi + torch.log(torch.tensor([self.gamma], device=device))
        )
        self.register_buffer("logA_x", logA_x)
        self.register_buffer("logA_y", logA_y)

    def forward(self, x, y):
        """
        x: array with shape (batch_size, x_dim)
        y: array with shape (batch_size, x_dim)
        """

        # Diffs for x (batch_size, num_s, x_dim)
        dx = x[:, None, :] - self.x_s[None, :, :]

        # Diffs for y (batch_size, num_s, y_dim)
        dy = y[:, None, :] - self.y_s[None, :, :]

        # Sum of square errors for x (summing over x_dim)
        sdx = ((dx / self.var_s) * dx).sum(dim=-1)

        # Sum of square errors for y (summing over y_dim)
        sdy = ((dy / self.gamma) * dy).sum(dim=-1)

        # Compute log p(x, y) under each sampled point in the mixture distribution
        # resulting tensor has shape (batch_size, num_s)
        log_pxy_each = -0.5 * (self.logA_x + self.logA_y + sdx + sdy)

        # Average over num_s (axis=-1)
        E_log_pxy = torch.logsumexp(log_pxy_each, axis=-1)

        # Compute log p(x) under each sampled point in the mixture distribution
        # resulting tensor has shape (batch_size, num_s)
        log_px_each = -0.5 * (self.logA_x + sdx)

        # Average over num_s (axis=-1)
        E_log_px = torch.logsumexp(log_px_each, axis=-1)

        # Compute log p(y) under each sampled point in the mixture distribution
        # resulting tensor has shape (batch_size, num_samples)
        log_py_each = -0.5 * (self.logA_y + sdy)

        # Average over num_samples (axis=-1)
        E_log_py = torch.logsumexp(log_py_each, axis=-1)

        return (E_log_pxy - E_log_px - E_log_py).mean()  # - 1/num_samples


class RecursiveLeastSquares(nn.Module):
    def __init__(
        self,
        nx,  # latent_dim
        ny,  # y_dim
        bias=False,
        polynomial_order=1,
    ):
        super().__init__()
        self.bias = bias
        self.polynomial_order = polynomial_order
        nx_poly = 0
        for i in range(1, polynomial_order + 1):
            nx_poly += torch.prod(torch.arange(nx, nx + i)) / torch.prod(
                torch.arange(i) + 1
            )
        nx = int(nx_poly) + bias
        self.register_buffer("theta", torch.zeros(nx, ny, requires_grad=False))
        self.register_buffer("P", torch.eye(nx, requires_grad=False))

        # Forgetting factor for the first linear decoder
        self.register_buffer("lam0", torch.tensor([0.9], requires_grad=False))

    def polynomial_expansion(self, x):
        """
        Parameters
        ----------
        x1 : torch.tensor
            (batch_size, num_features) matrix

        x2 : torch.tensor
            (batch_size, num_features) matrix

        Returns
        -------
        Z : torch.tensor
            (batch_size, num_quadratic_features) matrix.

        Note
        -----
        num_quadratic_features = num_features * (num_features + 1) / 2
        """
        n_features = x.shape[-1]
        x_list = [x]
        idx = torch.arange(x.shape[1], dtype=torch.long, device=x.device)
        for i in range(1, self.polynomial_order):
            C_idx = torch.combinations(idx, i + 1, with_replacement=True)
            x_list += [x[:, C_idx].prod(dim=-1) / len(C_idx) * n_features]

        return torch.column_stack(x_list)

    def update(self, x, y):
        # x (batch_size, latent_dim)
        # y (batch_size, y_dim)
        x = self.polynomial_expansion(x)
        if self.bias:
            x = torch.column_stack((x, torch.ones(x.shape[0], 1, device=x.device)))

        A = x @ self.P @ x.T
        A = torch.diagonal_scatter(
            A, A.diagonal(dim1=-2, dim2=-1) + self.lam0, dim1=-2, dim2=-1
        )
        self.P -= self.P @ x.T @ torch.linalg.solve(A, x @ self.P)
        self.P /= self.lam0
        self.theta = self.theta + self.P @ x.T @ (y - x @ self.theta)

    def forward(self, x):
        x = self.polynomial_expansion(x)
        if self.bias:
            x = torch.column_stack((x, torch.ones(x.shape[0], 1, device=x.device)))
        return x @ self.theta


class MovingAvgLeastSquares(nn.Module):
    def __init__(
        self,
        nx,
        ny,
        lamdiff=1e-1,
        delta=1e-4,
        bias=False,
        polynomial_order=1,
        l2_reg=0,
    ):
        super().__init__()
        self.bias = bias
        self.polynomial_order = polynomial_order
        nx_poly = 0
        for i in range(1, polynomial_order + 1):
            nx_poly += torch.prod(torch.arange(nx, nx + i)) / torch.prod(
                torch.arange(i) + 1
            )

        nx = int(nx_poly) + self.bias
        if l2_reg == None:
            self.l2_reg = 0
        else:
            self.l2_reg = l2_reg

        print("Moving Avg Least Squares Bias: {}".format(self.bias))
        # Running average of covariances for first linear decoder
        self.register_buffer("Sxx0", torch.eye(nx, requires_grad=False))
        self.register_buffer("Sxy0", torch.zeros(nx, ny, requires_grad=False))

        # Running average of covariances for first linear decoder
        self.register_buffer("Sxx1", torch.eye(nx, requires_grad=False))
        self.register_buffer("Sxy1", torch.zeros(nx, ny, requires_grad=False))

        # Forgetting factor for the first linear decoder
        self.register_buffer("lam0", torch.tensor([0.9], requires_grad=False))

        # Forgetting factor for the second linear decoder
        self.register_buffer("lam1", self.lam0 + lamdiff)

        # Update parameters for the forgetting factors
        self.delta = delta
        self.lamdiff = lamdiff

    def polynomial_expansion(self, x):
        """
        Parameters
        ----------
        x1 : torch.tensor
            (batch_size, num_features) matrix

        x2 : torch.tensor
            (batch_size, num_features) matrix

        Returns
        -------
        Z : torch.tensor
            (batch_size, num_quadratic_features) matrix.

        Note
        -----
        num_quadratic_features = num_features * (num_features + 1) / 2
        """
        n_features = x.shape[-1]
        x_list = [x]
        idx = torch.arange(x.shape[1], dtype=torch.long, device=x.device)
        for i in range(1, self.polynomial_order):
            C_idx = torch.combinations(idx, i + 1, with_replacement=True)
            x_list += [x[:, C_idx].prod(dim=-1) / len(C_idx) * n_features]

        return torch.column_stack(x_list)

    def forward(self, x):
        x = self.polynomial_expansion(x)

        if self.bias:
            x = torch.column_stack((x, torch.ones(x.shape[0], 1, device=x.device)))
            l2_reg = torch.ones(x.shape[1], device=x.device) * self.l2_reg
            l2_reg[-1] = 0
        else:
            l2_reg = torch.ones(x.shape[1], device=x.device) * self.l2_reg

        # Solve optimal decoder weights (normal equations)
        # import pdb; pdb.set_trace()
        W0 = torch.linalg.solve(
            self.Sxx0.diagonal_scatter(self.Sxx0.diagonal() + l2_reg), self.Sxy0
        )
        W1 = torch.linalg.solve(
            self.Sxx1.diagonal_scatter(self.Sxx1.diagonal() + l2_reg), self.Sxy1
        )

        # Predicted values for y
        yhat0 = x @ W0
        yhat1 = x @ W1
        return [yhat0, yhat1]

    def update(self, x, y):
        x = self.polynomial_expansion(x)

        if self.bias:
            x = torch.column_stack((x, torch.ones(x.shape[0], 1, device="cuda")))
        xx = (x.T @ x).detach()
        xy = (x.T @ y).detach()

        # Compute moving averages for the next batch of data
        self.Sxx0 = self.lam0 * self.Sxx0 + xx
        self.Sxy0 = self.lam0 * self.Sxy0 + xy
        self.Sxx1 = self.lam1 * self.Sxx1 + xx
        self.Sxy1 = self.lam1 * self.Sxy1 + xy
        return self

    def evaluate_loss(self, yhat0, yhat1, y):
        """
        Parameters
        ----------
        x : torch.tensor
            (batch_size x nx) matrix of independent variables.

        y : torch.tensor
            (batch_size x ny) matrix of dependent variables.

        Returns
        -------
        loss : torch.tensor
            Scalar loss reflecting average mean squared error of the
            two moving average estimates of the linear decoder.
        """
        # Loss for each decoder
        l0 = mse_loss(y, yhat0, reduction="sum")
        l1 = mse_loss(y, yhat1, reduction="sum")

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
    def __init__(self, in_dim, out_dim, bound=False):
        super(MLPEnsemble, self).__init__()

        # self.lin = nn.Sequential(
        #     nn.Linear(in_dim, out_dim),
        #     # nn.Tanh() if bound else None,
        # )

        self.mlp1 = nn.Sequential(
            nn.Linear(in_dim, in_dim),
            nn.ReLU(),
            nn.Linear(in_dim, in_dim),
            nn.ReLU(),
            nn.Linear(in_dim, out_dim),
            # nn.Tanh() if bound else None,
        )

        self.mlp2 = nn.Sequential(
            nn.Linear(in_dim, in_dim),
            nn.ReLU(),
            nn.Linear(in_dim, out_dim),
            # nn.Tanh() if bound else None,
        )

        self.mlp3 = nn.Sequential(
            nn.Linear(in_dim, in_dim),
            nn.ReLU(),
            nn.Linear(in_dim, in_dim // 2),
            nn.ReLU(),
            nn.Linear(in_dim // 2, out_dim),
            # nn.Tanh() if bound else None,
        )

        self.mlp4 = nn.Sequential(
            nn.Linear(in_dim, in_dim * 2),
            nn.ReLU(),
            nn.Linear(in_dim * 2, in_dim * 2),
            nn.ReLU(),
            nn.Linear(in_dim * 2, out_dim),
            # nn.Tanh() if bound else None,
        )

    def forward(self, z):
        # a = self.lin(z)
        a = self.mlp1(z)
        b = self.mlp2(z)
        c = self.mlp3(z)
        d = self.mlp4(z)
        return [a, b, c, d]  # a,


class GRScrubber(nn.Module):
    def __init__(self, in_dim, out_dim, alpha=1.0, bound=False):
        super(GRScrubber, self).__init__()
        self.reversal = nn.Sequential(
            GradientReversalLayer(alpha), MLPEnsemble(in_dim, out_dim, bound)
        )

    def forward(self, z):
        return self.reversal(z)

    def reset_parameters(self):
        for head in self.reversal[1].mlp1:
            if isinstance(head, nn.Linear):
                head.reset_parameters()

        for head in self.reversal[1].mlp2:
            if isinstance(head, nn.Linear):
                head.reset_parameters()

        for head in self.reversal[1].mlp3:
            if isinstance(head, nn.Linear):
                head.reset_parameters()

        for head in self.reversal[1].mlp4:
            if isinstance(head, nn.Linear):
                head.reset_parameters()


class AdvNetScrubber(nn.Module):
    def __init__(self, in_dim):
        super(AdvNetScrubber, self).__init__()
        self.ensemble = MLPEnsemble(in_dim, 2, False)
        self.optimizer = optim.AdamW(self.parameters(), lr=0.1)
        self.soft_max = nn.Softmax(-1)

        for param in self.parameters():
            param.requires_grad = False

    def forward(self, z, v):
        z = torch.cat([z, v], dim=-1)
        y = [self.soft_max(y_i) for y_i in self.ensemble(z)]
        return y

    def shuffle(self, z, v, v_ind):
        v_shuffle = v.clone()
        v_shuffle[:, v_ind] = v[torch.randperm(len(z)), v_ind]
        v_aug = torch.cat([v, v_shuffle], dim=0)
        z_aug = z.repeat(2, 1)

        return z_aug, v_aug

    def fit(self, z, v, v_ind, y=None, n_iter=5):
        for param in self.parameters():
            param.requires_grad = True
        if y is None:
            y = (
                torch.tensor([0, 1], device=z.device)[:, None]
                .repeat(1, z.shape[0])
                .ravel()
            )
            y = nn.functional.one_hot(y, 2).float()

        with torch.enable_grad():
            for _ in range(n_iter):
                for param in self.parameters():
                    param.grad = None

                z_aug, v_aug = self.shuffle(z, v, v_ind)
                y_pred = self.forward(z_aug, v_aug)
                ce = nn.CrossEntropyLoss(reduction="sum")
                loss = 0
                for y_ens in y_pred:
                    loss += ce(y_ens, y)

                (loss / len(y_pred) / z.shape[0]).backward()
                self.optimizer.step()

        for param in self.parameters():
            param.requires_grad = False
        return self.eval()


class LinearProjection(nn.Module):
    def __init__(
        self,
        in_dim,
        out_dim,
        bias=False,
    ):
        super(LinearProjection, self).__init__()
        self.decoder = nn.Linear(in_dim, out_dim, bias=bias)

    def forward(self, z):
        x = self.decoder(z)
        w = self.decoder.weight

        nrm = w @ w.T
        z_null = z - torch.linalg.solve(nrm, x.T).T @ w
        data_o = {"v": x, "z_null": z_null}
        return data_o


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
                    GradientReversalLayer(alpha), MLPEnsemble(in_dim, out_dim)
                )
            else:
                self.reversal = nn.Sequential(
                    GradientReversalLayer(alpha), MLPEnsemble(in_dim, out_dim, n_models)
                )
        else:
            self.reversal = None

    def forward(self, z):
        x = self.decoder(z)
        w = self.decoder.weight

        nrm = w @ w.T
        z_null = z - torch.linalg.solve(nrm, x.T).T @ w

        data_o = {"v": x, "mu_null": z_null}

        if self.reversal is not None:
            data_o["gr"] = self.reversal(z_null)

        return data_o
