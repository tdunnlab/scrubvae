import torch


def get_beta_schedule(beta, n_epochs, beta_anneal=False, M=4, R=0.75):
    if beta_anneal:
        print("Cyclical beta anneal")
        cycle_len = n_epochs // M
        beta_increase = torch.linspace(0, beta ** (1 / 4), int(cycle_len * R)) ** 4
        beta_plateau = torch.ones(cycle_len - len(beta_increase)) * beta

        beta_schedule = torch.cat([beta_increase, beta_plateau]).repeat(M)

        if len(beta_schedule) < n_epochs:
            beta_schedule = torch.cat(
                [beta_schedule, torch.ones(n_epochs - len(beta_schedule)) * beta]
            )
    else:
        print("No beta anneal")
        beta_schedule = torch.ones(n_epochs) * beta

    print(beta_schedule)

    return beta_schedule
