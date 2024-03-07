import os
import torch

from dnn.ak_sample_loader import AkSampleLoader
from dnn.torch_dataset import makeDataLoader, makeTrainingSample
from dnn.training import Trainer
from dnn.validation import inferenceOnSavedModel, doFullValidation

outputPath = "/grid_mnt/data_cms_upgrade/cuisset/ticlRegression/models/v2"

from dnn.model import BasicDNN, loss_mse_basic, BigDNN
device = torch.device('cuda:0')



def doTrainVal(tag, model, optimizer):
    input = AkSampleLoader.loadFromPickle("/grid_mnt/data_cms_upgrade/cuisset/ticlRegression/PionSamples_v1/fullData.pkl")
    train_dataloader = makeDataLoader(makeTrainingSample(input), batch_size=1000, num_workers=10)
    model.to(device)
    trainer = Trainer(model, loss_mse_basic, train_dataloader, optimizer, device)

    trainer.full_train(10)

    os.makedirs(outputPath + "/" + tag)
    trainer.save(outputPath + "/" + tag + "/model.pt")

    doFullValidation(outputPath + "/" + tag + "/model.pt", model, input)

# model = BasicDNN()
# optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-3)
# doTrainVal("basic", model, optimizer)

model = BigDNN()
optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-3)
doTrainVal("bigdnn-workers4", model, optimizer)