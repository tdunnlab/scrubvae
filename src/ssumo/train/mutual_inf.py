import torch

class MutInfoEstimator(torch.nn.Module):

    def __init__(self, x_s, y_s, var_s, bandwidth, var_mode="sphere", device="cuda"):
        """
        x_s: array with shape (num_s, x_dim)
        y_s: array with shape (num_s, y_dim)
        var_s: float or array/tensor (num_s, x_dim)
        gamma: float
        device: "cuda" or "cpu"
        """
        super().__init__()
        self.register_buffer("x_s", x_s)
        self.register_buffer("y_s",y_s)
        log2pi = torch.log(torch.tensor([2 * torch.pi], device=device))
        self.num_s = x_s.shape[0]
        assert y_s.shape[0] == self.num_s
        self.x_dim = x_s.shape[1]
        self.y_dim = y_s.shape[1]
        self.var_mode = var_mode

        if self.var_mode == "sphere":
            self.register_buffer("var_s",torch.tensor([var_s], device=device, requires_grad=False))
            logA_x = self.x_dim * (log2pi + torch.log(self.var_s))
        elif self.var_mode == "diagonal":
            self.register_buffer("var_s", var_s)
            logA_x = (
                self.x_dim * log2pi + torch.sum(torch.log(self.var_s), dim=-1)
            )[None, :]

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

        return (E_log_pxy - E_log_px - E_log_py).mean() # - 1/num_samples
 