from torch import nn
import torch
from torch_scatter import scatter_add
import numpy as np
from lightning.pytorch.core.mixins import HyperparametersMixin

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
            nn.Linear(nfeat, 1),
        )

class LinearEnergyCellTypeOnlyModel(nn.Module):
    def __init__(self, nfeat=10):
        super().__init__()
        self.model = nn.Sequential(
            nn.Linear(7, 1),
        )

    def forward(self, x):
        return torch.squeeze(self.model(x[:, 3:]), dim=1)

class BabyDNN(BasePredModel):
    def __init__(self, nfeat=10):
        super().__init__()
        self.model = nn.Sequential(
            nn.Linear(nfeat, 3),
            nn.ReLU(),
            nn.Linear(3, 1)
        )

class BasicDNN(BasePredModel):
    def __init__(self, nfeat=10):
        super().__init__()
        self.model = nn.Sequential(
            nn.Linear(nfeat, 10),
            nn.ReLU(),
            nn.Linear(10, 1),
        )

class MediumDNN(BasePredModel):
    def __init__(self, nfeat=10):
        super().__init__()
        self.model = nn.Sequential(
            nn.Linear(nfeat, 20),
            nn.ReLU(),
            nn.Linear(20, 20),
            nn.ReLU(),
            nn.Linear(20, 10),
            nn.ReLU(),
            nn.Linear(10, 1),
        )

class LargeDNN(BasePredModel):
    def __init__(self, nfeat=10):
        super().__init__()
        self.model = nn.Sequential(
            nn.Linear(nfeat, 30),
            nn.ReLU(),
            nn.Linear(30, 50),
            nn.ReLU(),
            nn.Linear(50, 50),
            nn.ReLU(),
            nn.Linear(50, 40),
            nn.ReLU(),
            nn.Linear(40, 30),
            nn.ReLU(),
            nn.Linear(30, 20),
            nn.ReLU(),
            nn.Linear(20, 10),
            nn.ReLU(),
            nn.Linear(10, 5),
            nn.ReLU(),
            nn.Linear(5, 1),
        )

class ParametrizedDNN(BasePredModel, HyperparametersMixin):
    def __init__(self, nfeat, hidden_size, num_layers):
        super().__init__()
        model_seq = [nn.Linear(nfeat, hidden_size), nn.ReLU()]
        for i in range(num_layers):
            model_seq.append(nn.Linear(hidden_size, hidden_size))
            model_seq.append(nn.ReLU())
        model_seq.append(nn.Linear(hidden_size, 1))
        self.model = nn.Sequential(*model_seq)
        self.save_hyperparameters()



def endcap_sum_predictions(model, data_batch):
    """ Sum the model predictions for an endcap """
    return scatter_add(model(data_batch["features"]), data_batch["tracksterInEvent_idx"])

def loss_mse_basic(model, data_batch):
    """ MSE of sumPredictions - CPenergy"""
    return nn.functional.mse_loss(endcap_sum_predictions(model, data_batch), data_batch["cp_energy"])

def loss_mse_basic_ratio(model, data_batch):
    """ MSE of sumPredictions/CPenergy - 1
    L = \sum_{events} \left(  \frac{\sum_{i \in \text{tracksters}} DNN(\text{features}_i)}{E_{\text{CaloParticle}}} -1\right )^2
    """
    return nn.functional.mse_loss(endcap_sum_predictions(model, data_batch)/data_batch["cp_energy"], torch.ones_like(data_batch["cp_energy"]))

def loss_mse_basic_ratio_constrained(model, data_batch, negative_regularization_coef=1., minFractionOfRawEnergy=0.9, minFractionOfRawEnergy_regCoeff=1.):
    """ MSE of sumPredictions/CPenergy - 1 with constraints
    L = \sum_{events} \left(  \frac{\sum_{i \in \text{tracksters}} DNN(\text{features}_i)}{E_{\text{CaloParticle}}} -1\right )^2
    """
    model_pred = model(data_batch["features"])
    sum_model_pred = scatter_add(model_pred, data_batch["tracksterInEvent_idx"])
    return (
        nn.functional.mse_loss(sum_model_pred/data_batch["cp_energy"], torch.ones_like(data_batch["cp_energy"])) 
        + negative_regularization_coef * nn.functional.relu(-model_pred).sum() # avoid negative energy prediction

        # avoid energy prediction being too far lower than raw_energy
        + minFractionOfRawEnergy_regCoeff* (nn.functional.relu(minFractionOfRawEnergy*data_batch["features"][:, 0] - model_pred)/data_batch["features"][:, 0]).sum()
    )



def getResultsFromModel_basicLoss(model:nn.Module, pred_batch:dict[str, torch.Tensor]) -> tuple[np.ndarray, np.ndarray]:
    """ Computes for the given model the energy prediction for each trackster as well as the sum of the energy predictions for the complete endcap """
    model_output_tensor = model(pred_batch["features"]).detach()
    full_energy_pred = scatter_add(model_output_tensor, pred_batch["tracksterInEvent_idx"].detach()).cpu().numpy()
    return model_output_tensor.cpu().numpy(), full_energy_pred


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

