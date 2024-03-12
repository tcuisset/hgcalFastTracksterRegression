""" Training using v2 samples (skeleton) pions with weighted samples 1/cp_energy and saving all fixed energy validation + fix NClusters variable """
import os
import torch
import matplotlib
matplotlib.use('Agg')
from functools import partial
import lightning as L
import lightning.pytorch.callbacks as L_c
from lightning.pytorch.loggers import TensorBoardLogger

from dnn.ak_sample_loader import AkSampleLoader, features
from dnn.torch_dataset import *
from dnn.training import Trainer
from dnn.validation import *

from dnn.train_lightning import *

outputPath = "/grid_mnt/data_cms_upgrade/cuisset/ticlRegression/models/v10"
feature_version = "feat-v2"
nfeat = len(features[feature_version])

from dnn.model import *
device_l = [0]
device = "cuda:0"

input = AkSampleLoader.loadFromPickle("/grid_mnt/data_cms_upgrade/cuisset/ticlRegression/PionSamples_v2/fullData.pkl")

input_test_perEnergy = {
    10 : '/grid_mnt/data_cms_upgrade/cuisset/ticlRegression/PionSamples_v2/FixedEnergy_10.pkl',
 200 : '/grid_mnt/data_cms_upgrade/cuisset/ticlRegression/PionSamples_v2/FixedEnergy_200.pkl',
 300 : '/grid_mnt/data_cms_upgrade/cuisset/ticlRegression/PionSamples_v2/FixedEnergy_300.pkl',
 400: '/grid_mnt/data_cms_upgrade/cuisset/ticlRegression/PionSamples_v2/FixedEnergy_400.pkl',
 600 : '/grid_mnt/data_cms_upgrade/cuisset/ticlRegression/PionSamples_v2/FixedEnergy_600.pkl'}
input_test_perEnergy = {energy : AkSampleLoader.loadFromPickle(inputFile) for energy, inputFile in input_test_perEnergy.items()}
datasets_perEnergy = {energy : RegressionDataset(input, feature_version, device=device) for energy, input in input_test_perEnergy.items()}

energies_PU = [10, 20, 50, 100, 200, 500]
input_PU_perEnergy = {energy : AkSampleLoader.loadFromPickle(f"/grid_mnt/data_cms_upgrade/cuisset/ticlRegression/PionSamples_PU/v1/{energy}.pkl") for energy in energies_PU}
datasets_PU = {energy : RegressionDataset(input, feature_version, device=device) for energy, input in input_PU_perEnergy.items()}


loss_constrained_v1 = partial(loss_mse_basic_ratio_constrained, negative_regularization_coef=0.1, minFractionOfRawEnergy=0.9, minFractionOfRawEnergy_regCoeff=5.)


l_model = FastDNNModule(loss_constrained_v1, getResultsFromModel_basicLoss, datasets_perEnergy, datasets_PU)

trainer = L.Trainer(max_epochs=10,
                    logger=TensorBoardLogger("/grid_mnt/data_cms_upgrade/cuisset/ticlRegression/models/v11"), enable_progress_bar=True, devices=device_l,
                    callbacks=[L_c.EarlyStopping("validation_loss")])
trainer.fit(model=l_model, train_dataloaders=makeDataLoader(makeTrainingSample(input, feature_version, device=device),  weighted=True, batch_size=1000, num_workers=0),
            val_dataloaders=makeDataLoader(makeValidationSample(input, feature_version, device=device), batch_size=10000))


trainer.test(model=l_model, dataloaders=[makeDataLoader(dataset, batch_size=10000) for dataset in datasets_perEnergy.values()] + [makeDataLoader(dataset, batch_size=10000) for dataset in datasets_PU.values()])