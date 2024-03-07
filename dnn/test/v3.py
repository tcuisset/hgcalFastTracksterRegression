""" Training using v2 samples (skeleton) pions"""
import os
import torch

from dnn.ak_sample_loader import AkSampleLoader
from dnn.torch_dataset import makeDataLoader, makeTrainingSample
from dnn.training import Trainer
from dnn.validation import inferenceOnSavedModel, doFullValidation

outputPath = "/grid_mnt/data_cms_upgrade/cuisset/ticlRegression/models/v3"

from dnn.model import *
device = torch.device('cuda:0')



def doTrainVal(tag, model, optimizer, nepochs=30):
    input = AkSampleLoader.loadFromPickle("/grid_mnt/data_cms_upgrade/cuisset/ticlRegression/PionSamples_v2/fullData.pkl")
    train_dataloader = makeDataLoader(makeTrainingSample(input), batch_size=1000, num_workers=10)
    model.to(device)
    trainer = Trainer(model, loss_mse_basic, train_dataloader, optimizer, device)

    trainer.full_train(nepochs)

    os.makedirs(outputPath + "/" + tag)
    trainer.save(outputPath + "/" + tag + "/model.pt")

    doFullValidation(outputPath + "/" + tag + "/model.pt", model, input)

# model = LinearModel()
# optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-3)
# doTrainVal("linear-v3", model, optimizer, nepochs=50)

model = LinearEnergyCellTypeOnlyModel()
optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-3)
doTrainVal("linear-onlyCellType", model, optimizer, nepochs=50)

# model = BabyDNN()
# optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4, weight_decay=1e-3)
# doTrainVal("babyDNN", model, optimizer)

# model = MediumDNN()
# optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4, weight_decay=1e-3)
# doTrainVal("mediumDNN", model, optimizer)

# model = LargeDNN()
# optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4, weight_decay=1e-3)
# doTrainVal("largeDNN", model, optimizer)

