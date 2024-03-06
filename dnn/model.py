from torch import nn
import torch
from torch_scatter import scatter_add


class BasicDNN(nn.Module):
    def __init__(self):
        super().__init__()
        self.model = nn.Sequential(
            nn.Linear(10, 10, dtype=torch.float64),
            nn.Linear(10, 1, dtype=torch.float64),
        )

    def forward(self, x):
        return torch.squeeze(self.model(x), dim=1)

def endcap_energy_pred(model, data_batch):
    return scatter_add(model(data_batch["features"]), data_batch["tracksterInEvent_idx"])


def loss_mse_basic(model, data_batch):
    return nn.functional.mse_loss(endcap_energy_pred(model, data_batch), data_batch["cp_energy"])