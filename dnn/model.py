from torch import nn
import torch
from torch_scatter import scatter_add
import numpy as np

from dnn.ak_sample_loader import FEATURES_INDICES

class BasePredModel(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, x):
        return torch.squeeze(self.model(x), dim=1)
        
class LinearModel(BasePredModel):
    def __init__(self, nfeat=10):
        super().__init__()
        self.model = nn.Sequential(
            nn.Linear(nfeat, 1, dtype=torch.float64),
        )

class LinearEnergyCellTypeOnlyModel(nn.Module):
    def __init__(self, nfeat=10):
        super().__init__()
        self.model = nn.Sequential(
            nn.Linear(7, 1, dtype=torch.float64),
        )

    def forward(self, x):
        return torch.squeeze(self.model(x[:, 3:]), dim=1)

class BabyDNN(BasePredModel):
    def __init__(self, nfeat=10):
        super().__init__()
        self.model = nn.Sequential(
            nn.Linear(nfeat, 3, dtype=torch.float64),
            nn.ReLU(),
            nn.Linear(3, 1, dtype=torch.float64)
        )

class BasicDNN(BasePredModel):
    def __init__(self, nfeat=10):
        super().__init__()
        self.model = nn.Sequential(
            nn.Linear(nfeat, 10, dtype=torch.float64),
            nn.ReLU(),
            nn.Linear(10, 1, dtype=torch.float64),
        )

class MediumDNN(BasePredModel):
    def __init__(self, nfeat=10):
        super().__init__()
        self.model = nn.Sequential(
            nn.Linear(nfeat, 20, dtype=torch.float64),
            nn.ReLU(),
            nn.Linear(20, 20, dtype=torch.float64),
            nn.ReLU(),
            nn.Linear(20, 10, dtype=torch.float64),
            nn.ReLU(),
            nn.Linear(10, 1, dtype=torch.float64),
        )

class LargeDNN(BasePredModel):
    def __init__(self, nfeat=10):
        super().__init__()
        self.model = nn.Sequential(
            nn.Linear(nfeat, 30, dtype=torch.float64),
            nn.ReLU(),
            nn.Linear(30, 50, dtype=torch.float64),
            nn.ReLU(),
            nn.Linear(50, 50, dtype=torch.float64),
            nn.ReLU(),
            nn.Linear(50, 40, dtype=torch.float64),
            nn.ReLU(),
            nn.Linear(40, 30, dtype=torch.float64),
            nn.ReLU(),
            nn.Linear(30, 20, dtype=torch.float64),
            nn.ReLU(),
            nn.Linear(20, 10, dtype=torch.float64),
            nn.ReLU(),
            nn.Linear(10, 5, dtype=torch.float64),
            nn.ReLU(),
            nn.Linear(5, 1, dtype=torch.float64),
        )


def endcap_sum_predictions(model, data_batch):
    """ Sum the model predictions for an endcap """
    return scatter_add(model(data_batch["features"]), data_batch["tracksterInEvent_idx"])

def loss_mse_basic(model, data_batch):
    """ MSE of sumPredictions - CPenergy"""
    return nn.functional.mse_loss(endcap_sum_predictions(model, data_batch), data_batch["cp_energy"])

def loss_mse_basic_ratio(model, data_batch):
    """ MSE of sumPredictions/CPenergy - 1"""
    return nn.functional.mse_loss(endcap_sum_predictions(model, data_batch)/data_batch["cp_energy"], torch.ones_like(data_batch["cp_energy"]))


def getResultsFromModel_basicLoss(model:nn.Module, pred_batch:dict[str, torch.Tensor]) -> tuple[np.ndarray, np.ndarray]:
    """ Computes for the given model the energy prediction for each trackster as well as the sum of the energy predictions for the complete endcap """
    model_output_tensor = model(pred_batch["features"]).detach()
    full_energy_pred = scatter_add(model_output_tensor, pred_batch["tracksterInEvent_idx"].detach()).numpy()
    return model_output_tensor.numpy(), full_energy_pred


def endcap_weightedSum_predictions(model, data_batch):
    """ Sum model predictions for an endcap, weighting by raw trackster energy """
    return scatter_add(model(data_batch["features"]) * data_batch["features"][:, FEATURES_INDICES.RAW_ENERGY], data_batch["tracksterInEvent_idx"])

def loss_mse_fractionPrediction(model, data_batch):
    """ MSE for the network predicting correction factor to raw trackster energy
    We want the network to predict a correction factor to the trackster energy, such that pred*tsEnergy + ... -> caloParticleEnergy
    """
    return nn.functional.mse_loss(endcap_weightedSum_predictions(model, data_batch), data_batch["cp_energy"])

def getResultsFromModel_lossFractionPrediction(model:nn.Module, pred_batch:dict[str, torch.Tensor]) -> tuple[np.ndarray, np.ndarray]:
    """ Computes for the given model the energy prediction for each trackster as well as the sum of the energy predictions for the complete endcap """
    model_output_tensor = model(pred_batch["features"]).detach()
    full_energy_pred = scatter_add(model_output_tensor * pred_batch["features"][:, FEATURES_INDICES.RAW_ENERGY].detach(), pred_batch["tracksterInEvent_idx"].detach()).detach().numpy()
    return model_output_tensor.numpy(), full_energy_pred

def loss_mse_fractionPrediction_ratio(model, data_batch):
    """ MSE for the network predicting correction factor to raw trackster energy, as ratio
    """
    return nn.functional.mse_loss(endcap_weightedSum_predictions(model, data_batch)/data_batch["cp_energy"], torch.ones_like(data_batch["cp_energy"]))

