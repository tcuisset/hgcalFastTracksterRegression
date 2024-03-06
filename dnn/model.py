from torch import nn
import torch
from torch_scatter import scatter_add

from dnn.ak_sample_loader import FEATURES_INDICES

class BasicDNN(nn.Module):
    def __init__(self):
        super().__init__()
        self.model = nn.Sequential(
            nn.Linear(10, 10, dtype=torch.float64),
            nn.Linear(10, 1, dtype=torch.float64),
        )

    def forward(self, x):
        return torch.squeeze(self.model(x), dim=1)

def endcap_sum_predictions(model, data_batch):
    """ Sum the model predictions for an endcap """
    return scatter_add(model(data_batch["features"]), data_batch["tracksterInEvent_idx"])

def endcap_weightedSum_predictions(model, data_batch):
    """ Sum model predictions for an endcap, weighting by raw trackster energy """
    return scatter_add(model(data_batch["features"]) * data_batch["features"][:, FEATURES_INDICES.RAW_ENERGY], data_batch["tracksterInEvent_idx"])

def loss_mse_basic(model, data_batch):
    """ MSE of sumPredictions - CPenergy"""
    return nn.functional.mse_loss(endcap_sum_predictions(model, data_batch), data_batch["cp_energy"])

def loss_mse_ratio(model, data_batch):
    """ MSE of sumPredictions*
    We want the network to predict a correction factor to the trackster energy, such that pred*tsEnergy + ... -> caloParticleEnergy
    """
    

    return nn.functional.mse_loss(endcap_sum_predictions(model, data_batch), data_batch["cp_energy"])