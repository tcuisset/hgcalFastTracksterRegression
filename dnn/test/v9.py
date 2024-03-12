""" Training using v2 samples (skeleton) pions with weighted samples 1/cp_energy and saving all fixed energy validation + fix NClusters variable """
import os
import torch
import matplotlib
matplotlib.use('Agg')

from dnn.ak_sample_loader import AkSampleLoader, features
from dnn.torch_dataset import makeDataLoader, makeTrainingSample, RegressionDataset
from dnn.training import Trainer
from dnn.validation import inferenceOnSavedModel, doFullValidation, doFullValidation_fixedEnergy, doFullValidation_PU

outputPath = "/grid_mnt/data_cms_upgrade/cuisset/ticlRegression/models/v9"
feature_version = "feat-v2"
nfeat = len(features[feature_version])

from dnn.model import *
device = torch.device('cuda:0')

input = AkSampleLoader.loadFromPickle("/grid_mnt/data_cms_upgrade/cuisset/ticlRegression/PionSamples_v2/fullData.pkl")
input_test_perEnergy = {
    10 : '/grid_mnt/data_cms_upgrade/cuisset/ticlRegression/PionSamples_v2/FixedEnergy_10.pkl',
 200 : '/grid_mnt/data_cms_upgrade/cuisset/ticlRegression/PionSamples_v2/FixedEnergy_200.pkl',
 300 : '/grid_mnt/data_cms_upgrade/cuisset/ticlRegression/PionSamples_v2/FixedEnergy_300.pkl',
 400: '/grid_mnt/data_cms_upgrade/cuisset/ticlRegression/PionSamples_v2/FixedEnergy_400.pkl',
 600 : '/grid_mnt/data_cms_upgrade/cuisset/ticlRegression/PionSamples_v2/FixedEnergy_600.pkl'}
input_test_perEnergy = {energy : AkSampleLoader.loadFromPickle(inputFile) for energy, inputFile in input_test_perEnergy.items()}

energies_PU = [10, 20, 50, 100, 200, 500]
input_PU_perEnergy = {energy : AkSampleLoader.loadFromPickle(f"/grid_mnt/data_cms_upgrade/cuisset/ticlRegression/PionSamples_PU/v1/{energy}.pkl") for energy in energies_PU}
datasets_PU = {energy : RegressionDataset(input, feature_version, device=device) for energy, input in input_PU_perEnergy.items()}


def doTrainVal(tag, model, optimizer, scheduler=None, nepochs=100, loss=loss_mse_basic, resultsFromModel_fct=getResultsFromModel_basicLoss):
    train_dataloader = makeDataLoader(makeTrainingSample(input, feature_version), weighted=True, batch_size=1000, num_workers=25)
    model.to(device)
    trainer = Trainer(model, loss, train_dataloader, optimizer, scheduler, device)

    trainer.full_train(nepochs)

    os.makedirs(outputPath + "/" + tag)
    trainer.save(outputPath + "/" + tag + "/model.pt")

    doFullValidation(outputPath + "/" + tag + "/model.pt", model, input, feature_version, getResultsFromModel=resultsFromModel_fct)
    doFullValidation_fixedEnergy(outputPath + "/" + tag + "/model.pt", model, input_test_perEnergy, feature_version, getResultsFromModel=resultsFromModel_fct)
    doFullValidation_PU(outputPath + "/" + tag, model, input_PU_perEnergy, datasets_PU)


# model = LinearModel(nfeat)
# optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-3)
# doTrainVal("linear-v3", model, optimizer, nepochs=50)
def mkSched(opt, T_0=20):
    return torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(opt, T_0=T_0)

model = BabyDNN(nfeat)
optimizer = torch.optim.AdamW(model.parameters())
doTrainVal("babyDNN-ratioLoss-constrained", model, optimizer, mkSched(optimizer, 10), loss=loss_mse_basic_ratio_constrained, nepochs=40)

model = BabyDNN(nfeat)
optimizer = torch.optim.AdamW(model.parameters())
doTrainVal("babyDNN-ratioLoss", model, optimizer, mkSched(optimizer, 10), loss=loss_mse_basic_ratio, nepochs=40)


model = MediumDNN(nfeat)
optimizer = torch.optim.AdamW(model.parameters())
doTrainVal("mediumDNN-ratioLoss-constrained", model, optimizer, mkSched(optimizer), loss=loss_mse_basic_ratio_constrained)

model = MediumDNN(nfeat)
optimizer = torch.optim.AdamW(model.parameters())
doTrainVal("mediumDNN-ratioLoss", model, optimizer, mkSched(optimizer), loss=loss_mse_basic_ratio)


model = LargeDNN(nfeat)
optimizer = torch.optim.AdamW(model.parameters())
doTrainVal("largeDNN-ratioLoss-constrained", model, optimizer, mkSched(optimizer, 40), loss=loss_mse_basic_ratio_constrained, nepochs=200)

model = LargeDNN(nfeat)
optimizer = torch.optim.AdamW(model.parameters())
doTrainVal("largeDNN-ratioLoss", model, optimizer, mkSched(optimizer, 40), loss=loss_mse_basic_ratio, nepochs=200)





# model = BabyDNN(nfeat)
# optimizer = torch.optim.AdamW(model.parameters())
# doTrainVal("babyDNN-fractionPred", model, optimizer, mkSched(optimizer, 10), loss=loss_mse_fractionPrediction, resultsFromModel_fct=getResultsFromModel_lossFractionPrediction, nepochs=20)

# model = MediumDNN(nfeat)
# optimizer = torch.optim.AdamW(model.parameters())
# doTrainVal("mediumDNN-fractionPred", model, optimizer, mkSched(optimizer), loss=loss_mse_fractionPrediction, resultsFromModel_fct=getResultsFromModel_lossFractionPrediction, nepochs=70)

# model = LargeDNN(nfeat)
# optimizer = torch.optim.AdamW(model.parameters())
# doTrainVal("largeDNN-fractionPred", model, optimizer, mkSched(optimizer, 30), loss=loss_mse_fractionPrediction, resultsFromModel_fct=getResultsFromModel_lossFractionPrediction, nepochs=100)