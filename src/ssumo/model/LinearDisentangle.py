from torch.autograd import Function
import torch.nn as nn
import torch


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
    def __init__(self, in_dim, out_dim):
        super(ReversalEnsemble, self).__init__()

        self.lin = nn.Linear(in_dim, out_dim)

        self.mlp1 = nn.Sequential(
            nn.Linear(in_dim, in_dim),
            nn.ReLU(),
            nn.Linear(in_dim, in_dim),
            nn.ReLU(),
            nn.Linear(in_dim, out_dim),
        )

        self.mlp2 = nn.Sequential(
            nn.Linear(in_dim, in_dim), nn.ReLU(), nn.Linear(in_dim, out_dim)
        )

        self.mlp3 = nn.Sequential(
            nn.Linear(in_dim, in_dim),
            nn.ReLU(),
            nn.Linear(in_dim, in_dim // 2),
            nn.ReLU(),
            nn.Linear(in_dim // 2, out_dim),
        )

    def forward(self, z):
        a = self.lin(z)
        b = self.mlp1(z)
        c = self.mlp2(z)
        d = self.mlp3(z)
        return [a, b, c, d]

class Scrubber(nn.Module):
    def __init__(self, in_dim, out_dim, alpha=1.0):
        super(Scrubber, self).__init__()
        self.reversal = nn.Sequential(
            GradientReversalLayer(alpha), ReversalEnsemble(in_dim, out_dim)
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
