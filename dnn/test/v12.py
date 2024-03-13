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

##############
log_path = "/grid_mnt/data_cms_upgrade/cuisset/ticlRegression/models/v12c"
os.makedirs(log_path, exist_ok=True)

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


energies_PU = [10, 20, 50, 100, 200, 500]
input_PU_perEnergy = {energy : AkSampleLoader.loadFromPickle(f"/grid_mnt/data_cms_upgrade/cuisset/ticlRegression/PionSamples_PU/v1/{energy}_assoc_lowStat.pkl") for energy in energies_PU}

def train_fct(config):
    datasets_perEnergy = {energy : RegressionDataset(input, feature_version, device=device) for energy, input in input_test_perEnergy.items()}
    datasets_PU = {energy : RegressionDataset(input, feature_version, device=device) for energy, input in input_PU_perEnergy.items()}


    loss_constrained_v1 = partial(loss_mse_basic_ratio_constrained, negative_regularization_coef=config["negative_regularization_coef"], minFractionOfRawEnergy=config["minFractionOfRawEnergy"], minFractionOfRawEnergy_regCoeff=config["minFractionOfRawEnergy_regCoeff"])

    l_model = FastDNNModule(loss_constrained_v1, getResultsFromModel_basicLoss, datasets_perEnergy, datasets_PU, model=ParametrizedDNN(17, **{key[len("model."):]:val for key, val in config.items() if key.startswith("model.")}), **config)

    trainer = L.Trainer(max_epochs=300, default_root_dir=log_path,
                        logger=TensorBoardLogger(log_path), enable_progress_bar=False, devices=device_l,
                        log_every_n_steps=1,
                        callbacks=[L_c.EarlyStopping("validation_loss", patience=15, min_delta=0.02), L_c.ModelCheckpoint(save_top_k=5, monitor="validation_loss", every_n_epochs=3),
                                   L_c.LearningRateMonitor(logging_interval="epoch")],
                        )
    trainer.fit(model=l_model, train_dataloaders=makeDataLoader(makeTrainingSample(input, feature_version, device=device),  weighted=True, batch_size=config["batch_size"], num_workers=0),
                val_dataloaders=makeDataLoader(makeValidationSample(input, feature_version, device=device), batch_size=10000))


    trainer.test(model=l_model, dataloaders=[makeDataLoader(dataset, batch_size=10000) for dataset in datasets_perEnergy.values()] + [makeDataLoader(dataset, batch_size=10000) for dataset in datasets_PU.values()])

default_config = {
    "negative_regularization_coef" : 0.1, "minFractionOfRawEnergy" : 0.9, "minFractionOfRawEnergy_regCoeff" : 5.,
    "model.hidden_size": 20, "model.num_layers": 10,
    "lr": 1e-3, "batch_size": 1024,
}

def doTrain(config):
    # try:
    train_fct(default_config|config)
    # except Exception as e:
    #     print(e)
    #     with open(log_path + "/error_log.txt", "a") as myfile:
    #         myfile.write(str(e))
    #     raise e

from concurrent.futures import ProcessPoolExecutor
import itertools
import random
if __name__ == "__main__":


    with ProcessPoolExecutor(max_workers=15) as exc:
        configs_individual = [
            [{}, {"batch_size" : 512}, {"batch_size" : 5000}],
            [{}, {"lr" : 1e-4}],
            [{"negative_regularization_coef" : 10., "minFractionOfRawEnergy" : 0.9, "minFractionOfRawEnergy_regCoeff" : 5.},
            {"negative_regularization_coef" : 1., "minFractionOfRawEnergy" : 0.9, "minFractionOfRawEnergy_regCoeff" : 50.},
            {}],
            
            [{},{ "model.hidden_size": 10, "model.num_layers": 5},{ "model.hidden_size": 30, "model.num_layers": 40}]
        ]

        def getAllConfigs():
            for elt in itertools.product(*configs_individual):
                config = {}
                for d in elt:
                    config.update(d)
                yield config
        configs = list(getAllConfigs())
        random.shuffle(configs)
        list(exc.map(doTrain, configs))
        # list(exc.map(doTrain, [
        #     {},
        #     {"lr" : 1e-4},
        #     {"negative_regularization_coef" : 10., "minFractionOfRawEnergy" : 0.9, "minFractionOfRawEnergy_regCoeff" : 5.,},
        #     {"negative_regularization_coef" : 1., "minFractionOfRawEnergy" : 0.9, "minFractionOfRawEnergy_regCoeff" : 50.,}
        # ]))
