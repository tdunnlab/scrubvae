import torch.optim as optim
import torch
from ssumo.train.mutual_inf import MutInfoEstimator
import matplotlib.pyplot as plt
import numpy as np

class CorrelatedGaussians(torch.nn.Module):
    def __init__(self, x_dim=128, y_dim=6, init_correlation = 1):
        super().__init__()
        self.correlation = torch.nn.Parameter(torch.ones(1)*init_correlation)
        # self.register_parameter("correlation", correlation)
        self.x_dim = x_dim
        self.y_dim = y_dim

    def forward(self, n_samples):
        z = torch.randn(n_samples, self.x_dim, device="cuda")
        x = z * self.correlation + (1 - self.correlation) * torch.randn_like(z)

        y = z[:, : self.y_dim] * self.correlation + (
            1 - self.correlation
        ) * torch.randn_like(z[:, : self.y_dim])
        return x, y

bandwidths = [1, 1.5, 2]
init_correlations = [0.5, 0.9, 1]
f = plt.figure(figsize=(10,5))
for bw in bandwidths:
    for ic in init_correlations:
        cgauss = CorrelatedGaussians(x_dim=128, y_dim=5, init_correlation=ic)
        cgauss.cuda()
        optimizer = optim.SGD(cgauss.parameters(), lr=0.1)
        correlation = []
        with torch.enable_grad():
            for i in range(50):
                for param in cgauss.parameters():
                    param.grad = None
                x, y = cgauss(n_samples=2048)

                if i > 0:
                    loss = mi_estimator(x, y)
                    loss.backward()
                    optimizer.step()
                    cgauss.correlation.data.clamp_(0.0,1.0)

                mi_estimator = MutInfoEstimator(
                    x.detach().clone(),
                    y.detach().clone(),
                    bw,
                    gamma=bw,
                    var_mode="sphere",
                    device="cuda",
                )
                correlation += [cgauss.correlation.item()]

        plt.plot(np.arange(50), correlation, label = "Bandwidth: {}, Init_corr: {}".format(bw, ic))
        plt.xlabel("Iteration")
        plt.ylabel("Correlation")

plt.legend()
plt.savefig("./mi_correlation_train_demo.png")


# class MutInfoEstimator(torch.nn.Module):

#     def __init__(self, x_samples, y_samples, bandwidth):
#         """
#         x_samples: array with shape (num_samples, num_dims_x)
#         y_samples: array with shape (num_samples, num_dims_y)
#         bandwidth: float
#         """
#         super().__init__()
#         self.x_samples = x_samples
#         self.y_samples = y_samples
#         self.bandwidth = bandwidth
#         self.num_samples = x_samples.shape[0]
#         assert y_samples.shape[0] == self.num_samples
#         self.num_dims_x = x_samples.shape[1]
#         self.num_dims_y = y_samples.shape[1]

#     def cuda(self):
#         self.x_samples = self.x_samples.cuda()
#         self.y_samples = self.y_samples.cuda()
#         return self

#     def forward(self, x, y):
#         """
#         x: array with shape (batch_size, num_dims_x)
#         y: array with shape (batch_size, num_dims_x)
#         """

#         # Diffs for x (batch_size, num_samples, num_dims_x)
#         dx = x[:, None, :] - self.x_samples[None, :, :]

#         # Diffs for y (batch_size, num_samples, num_dims_y)
#         dy = y[:, None, :] - self.y_samples[None, :, :]

#         # Sum of square errors for x (summing over num_dims_x)
#         sdx = -0.5 * torch.sum(dx**2, axis=-1) / self.bandwidth

#         # Sum of square errors for y (summing over num_dims_y)
#         sdy = -0.5 * torch.sum(dy**2, axis=-1) / self.bandwidth

#         # Compute log p(x, y) under each sampled point in the mixture distribution
#         # resulting tensor has shape (batch_size, num_samples)
#         log_pxy_each = (
#             -0.5 * (self.num_dims_x + self.num_dims_y) * np.log(2 * torch.pi)
#             - 0.5 * (self.num_dims_x + self.num_dims_y) * np.log(self.bandwidth)
#             + (sdx + sdy)
#         )

#         # Average over num_samples (axis=-1)
#         E_log_pxy = np.log(1 / self.num_samples) + torch.logsumexp(
#             log_pxy_each, axis=-1
#         )

#         # Compute log p(x) under each sampled point in the mixture distribution
#         # resulting tensor has shape (batch_size, num_samples)
#         log_px_each = (
#             -0.5 * self.num_dims_x * np.log(2 * torch.pi)
#             - 0.5 * self.num_dims_x * np.log(self.bandwidth)
#             + sdx
#         )

#         # Average over num_samples (axis=-1)
#         E_log_px = np.log(1 / self.num_samples) + torch.logsumexp(log_px_each, axis=-1)

#         # Compute log p(y) under each sampled point in the mixture distribution
#         # resulting tensor has shape (batch_size, num_samples)
#         log_py_each = (
#             -0.5 * self.num_dims_y * np.log(2 * torch.pi)
#             - 0.5 * self.num_dims_y * np.log(self.bandwidth)
#             + sdy
#         )

#         # Average over num_samples (axis=-1)
#         E_log_py = np.log(1 / self.num_samples) + torch.logsumexp(log_py_each, axis=-1)

#         return E_log_pxy - E_log_px - E_log_py


# if __name__ == "__main__":
#     num_samples = 2048
#     batch_size = 2048
#     num_dims_x = 128
#     num_dims_y = 6
#     # bandwidth = 0.2

#     # num_dims_x_list = [128]

#     for bandwidth in [1, 1.25, 1.5, 2, 2.5, 3]:
#         mi = []
#         for c in np.linspace(0, 1, 100):
#             z_samples = torch.randn(num_samples, num_dims_x, device="cuda")
#             x_samples = z_samples * c + (1 - c) * torch.randn(
#                 num_samples, num_dims_x, device="cuda"
#             )
#             y_samples = z_samples[:,:num_dims_y] * c + (1 - c) * torch.randn(
#                 num_samples, num_dims_y, device="cuda"
#             )

#             estimator = MutInfoEstimator(x_samples, y_samples, bandwidth)

#             z_batch = torch.randn(batch_size, num_dims_x, device="cuda")
#             x_batch = z_batch * c + (1 - c) * torch.randn(
#                 batch_size, num_dims_x, device="cuda"
#             )
#             y_batch = z_batch[:, :num_dims_y] * c + (1 - c) * torch.randn(
#                 batch_size, num_dims_y, device="cuda"
#             )
#             mi += [torch.mean(estimator(x_batch, y_batch)).detach().cpu().numpy()]

#             print("Mut Info Est: ", mi[-1])

#         plt.plot(np.linspace(0, 1, 100), mi, label=str(bandwidth))
#     plt.xlabel("Correlation")
#     plt.ylabel("MI")
#     plt.legend()
#     plt.savefig("./mi_demo.png")


# # if __name__ == "__main__":

# #     num_samples = 10000
# #     batch_size = 10000
# #     num_dims_x = 2
# #     num_dims_y = 3
# #     bandwidth = 0.1

# #     mi = []
# #     for num_dims_x in range(2,128,2):
# #         x_samples = torch.randn(num_samples, num_dims_x)
# #         y_samples = torch.randn(num_samples, num_dims_y)

# #         estimator = MutInfoEstimator(
# #             x_samples, y_samples, bandwidth
# #         )

# #         x_batch = torch.randn(batch_size, num_dims_x)
# #         y_batch = torch.randn(batch_size, num_dims_y)

# #         mi+=[torch.mean(estimator(x_batch, y_batch))]

# #         print("Mut Info Est: ", torch.mean(estimator(x_batch, y_batch)))

# #     plt.plot(np.arange(2,128,2,), mi)
# #     plt.xlabel("X Dim")
# #     plt.ylabel("MI")
# #     plt.savefig("./mi_demo.png")


# # if __name__ == "__main__":

# #     num_samples = 2048
# #     batch_size = 2048
# #     num_dims_x = 128
# #     num_dims_y = 6
# #     # bandwidth = 0.2

# #     # num_dims_x_list = [128]

# #     for bandwidth in [1, 1.25, 1.5, 2, 2.5, 3]:
# #         mi = []
# #         for c in np.linspace(0, 1, 100):
# #             z_samples = torch.randn(num_samples, num_dims_x, device="cuda")
# #             x_samples = z_samples * c + (1 - c) * torch.randn(
# #                 num_samples, num_dims_x, device="cuda"
# #             )
# #             y_samples = z_samples[:, :num_dims_y] * c + (1 - c) * torch.randn(
# #                 num_samples, num_dims_y, device="cuda"
# #             )

# #             estimator = MutInfoEstimator(x_samples, y_samples, bandwidth, bandwidth)

# #             z_batch = torch.randn(batch_size, num_dims_x, device="cuda")
# #             x_batch = z_batch * c + (1 - c) * torch.randn(
# #                 batch_size, num_dims_x, device="cuda"
# #             )
# #             y_batch = z_batch[:, :num_dims_y] * c + (1 - c) * torch.randn(
# #                 batch_size, num_dims_y, device="cuda"
# #             )
# #             mi += [estimator(x_batch, y_batch).detach().cpu().numpy()]

# #             print("Mut Info Est: ", mi[-1])

# #         plt.plot(np.linspace(0, 1, 100), mi, label=str(bandwidth))
# #     plt.xlabel("Correlation")
# #     plt.ylabel("MI")
# #     plt.legend()
# #     plt.savefig("./test_mci.png")
