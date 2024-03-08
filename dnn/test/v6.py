""" Training using v2 samples (skeleton) pions with weighted samples 1/cp_energy"""
import os
import torch

from dnn.ak_sample_loader import AkSampleLoader, features
from dnn.torch_dataset import makeDataLoader, makeTrainingSample
from dnn.training import Trainer
from dnn.validation import inferenceOnSavedModel, doFullValidation

outputPath = "/grid_mnt/data_cms_upgrade/cuisset/ticlRegression/models/v6"
feature_version = "feat-v2"
nfeat = len(features[feature_version])

from dnn.model import *
device = torch.device('cuda:0')



def doTrainVal(tag, model, optimizer, nepochs=50, loss=loss_mse_basic, resultsFromModel_fct=getResultsFromModel_basicLoss):
    input = AkSampleLoader.loadFromPickle("/grid_mnt/data_cms_upgrade/cuisset/ticlRegression/PionSamples_v2/fullData.pkl")
    train_dataloader = makeDataLoader(makeTrainingSample(input, feature_version), weighted=True, batch_size=1000, num_workers=10)
    model.to(device)
    trainer = Trainer(model, loss, train_dataloader, optimizer, device)

    trainer.full_train(nepochs)

    os.makedirs(outputPath + "/" + tag)
    trainer.save(outputPath + "/" + tag + "/model.pt")

    doFullValidation(outputPath + "/" + tag + "/model.pt", model, input, feature_version, getResultsFromModel=resultsFromModel_fct)


# model = LinearModel(nfeat)
# optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-3)
# doTrainVal("linear-v3", model, optimizer, nepochs=50)

model = BabyDNN(nfeat)
optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4, weight_decay=1e-3)
doTrainVal("babyDNN", model, optimizer)

model = MediumDNN(nfeat)
optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4, weight_decay=1e-3)
doTrainVal("mediumDNN", model, optimizer)

model = LargeDNN(nfeat)
optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4, weight_decay=1e-3)
doTrainVal("largeDNN", model, optimizer)




# model = LinearModel(nfeat)
# optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-3)
# doTrainVal("linear-fractionPred", model, optimizer, nepochs=50, loss=loss_mse_fractionPrediction, resultsFromModel_fct=getResultsFromModel_lossFractionPrediction)

model = BabyDNN(nfeat)
optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4, weight_decay=1e-3)
doTrainVal("babyDNN-fractionPred", model, optimizer, loss=loss_mse_fractionPrediction, resultsFromModel_fct=getResultsFromModel_lossFractionPrediction)

model = MediumDNN(nfeat)
optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4, weight_decay=1e-3)
doTrainVal("mediumDNN-fractionPred", model, optimizer, loss=loss_mse_fractionPrediction, resultsFromModel_fct=getResultsFromModel_lossFractionPrediction)

model = LargeDNN(nfeat)
optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4, weight_decay=1e-3)
doTrainVal("largeDNN-fractionPred", model, optimizer, loss=loss_mse_fractionPrediction, resultsFromModel_fct=getResultsFromModel_lossFractionPrediction)