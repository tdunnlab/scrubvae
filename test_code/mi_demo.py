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


# if __name__ == "__main__":

#     num_samples = 10000
#     batch_size = 10000
#     num_dims_x = 2
#     num_dims_y = 3
#     bandwidth = 0.1

#     mi = []
#     for num_dims_x in range(2,128,2):
#         x_samples = torch.randn(num_samples, num_dims_x)
#         y_samples = torch.randn(num_samples, num_dims_y)

#         estimator = MutInfoEstimator(
#             x_samples, y_samples, bandwidth
#         )

#         x_batch = torch.randn(batch_size, num_dims_x)
#         y_batch = torch.randn(batch_size, num_dims_y)

#         mi+=[torch.mean(estimator(x_batch, y_batch))]

#         print("Mut Info Est: ", torch.mean(estimator(x_batch, y_batch)))

#     plt.plot(np.arange(2,128,2,), mi)
#     plt.xlabel("X Dim")
#     plt.ylabel("MI")
#     plt.savefig("./mi_demo.png")


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
#             y_samples = z_samples[:, :num_dims_y] * c + (1 - c) * torch.randn(
#                 num_samples, num_dims_y, device="cuda"
#             )

#             estimator = MutInfoEstimator(x_samples, y_samples, bandwidth, bandwidth)

#             z_batch = torch.randn(batch_size, num_dims_x, device="cuda")
#             x_batch = z_batch * c + (1 - c) * torch.randn(
#                 batch_size, num_dims_x, device="cuda"
#             )
#             y_batch = z_batch[:, :num_dims_y] * c + (1 - c) * torch.randn(
#                 batch_size, num_dims_y, device="cuda"
#             )
#             mi += [estimator(x_batch, y_batch).detach().cpu().numpy()]

#             print("Mut Info Est: ", mi[-1])

#         plt.plot(np.linspace(0, 1, 100), mi, label=str(bandwidth))
#     plt.xlabel("Correlation")
#     plt.ylabel("MI")
#     plt.legend()
#     plt.savefig("./test_mci.png")
