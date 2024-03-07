from torch import nn
import torch
from torch_scatter import scatter_add

from dnn.ak_sample_loader import FEATURES_INDICES

class BasePredModel(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, x):
        return torch.squeeze(self.model(x), dim=1)
        
class LinearModel(BasePredModel):
    def __init__(self):
        super().__init__()
        self.model = nn.Sequential(
            nn.Linear(10, 1, dtype=torch.float64),
        )

class LinearEnergyCellTypeOnlyModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.model = nn.Sequential(
            nn.Linear(7, 1, dtype=torch.float64),
        )

    def forward(self, x):
        return torch.squeeze(self.model(x[:, 3:]), dim=1)

class BabyDNN(BasePredModel):
    def __init__(self):
        super().__init__()
        self.model = nn.Sequential(
            nn.Linear(10, 3, dtype=torch.float64),
            nn.Linear(3, 1, dtype=torch.float64),
        )

class BasicDNN(BasePredModel):
    def __init__(self):
        super().__init__()
        self.model = nn.Sequential(
            nn.Linear(10, 10, dtype=torch.float64),
            nn.Linear(10, 1, dtype=torch.float64),
        )

class MediumDNN(BasePredModel):
    def __init__(self):
        super().__init__()
        self.model = nn.Sequential(
            nn.Linear(10, 20, dtype=torch.float64),
            nn.Linear(20, 20, dtype=torch.float64),
            nn.Linear(20, 10, dtype=torch.float64),
            nn.Linear(10, 1, dtype=torch.float64),
        )

class LargeDNN(BasePredModel):
    def __init__(self):
        super().__init__()
        self.model = nn.Sequential(
            nn.Linear(10, 30, dtype=torch.float64),
            nn.Linear(30, 50, dtype=torch.float64),
            nn.Linear(50, 50, dtype=torch.float64),
            nn.Linear(50, 40, dtype=torch.float64),
            nn.Linear(40, 30, dtype=torch.float64),
            nn.Linear(30, 20, dtype=torch.float64),
            nn.Linear(20, 10, dtype=torch.float64),
            nn.Linear(10, 5, dtype=torch.float64),
            nn.Linear(5, 1, dtype=torch.float64),
        )


def endcap_sum_predictions(model, data_batch):
    """ Sum the model predictions for an endcap """
    return scatter_add(model(data_batch["features"]), data_batch["tracksterInEvent_idx"])

def loss_mse_basic(model, data_batch):
    """ MSE of sumPredictions - CPenergy"""
    return nn.functional.mse_loss(endcap_sum_predictions(model, data_batch), data_batch["cp_energy"])

def endcap_weightedSum_predictions(model, data_batch):
    """ Sum model predictions for an endcap, weighting by raw trackster energy """
    return scatter_add(model(data_batch["features"]) * data_batch["features"][:, FEATURES_INDICES.RAW_ENERGY], data_batch["tracksterInEvent_idx"])

def loss_mse_fractionPrediction(model, data_batch):
    """ MSE for the network predicting correction factor to raw trackster energy
    We want the network to predict a correction factor to the trackster energy, such that pred*tsEnergy + ... -> caloParticleEnergy
    """
    return nn.functional.mse_loss(endcap_weightedSum_predictions(model, data_batch), data_batch["cp_energy"])

def loss_mse_fractionPrediction_ratio(model, data_batch):
    """ MSE for the network predicting correction factor to raw trackster energy, as ratio
    """
    return nn.functional.mse_loss(endcap_weightedSum_predictions(model, data_batch)/data_batch["cp_energy"], torch.ones_like(data_batch["cp_energy"]))