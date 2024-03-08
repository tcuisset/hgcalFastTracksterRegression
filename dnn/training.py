import torch
from torch.utils.data import DataLoader
from torch import nn
from tqdm.auto import tqdm

class Trainer:
    def __init__(self, model:nn.Module, loss_fn, dataloader:DataLoader, optimizer, scheduler=None, device="cpu") -> None:
        self.model = model
        self.loss_fn = loss_fn
        self.dataloader = dataloader
        self.optimizer = optimizer
        self.scheduler = scheduler
        self.device = device

        self.losses_per_epoch = []
        self.losses_per_batch = []

    def train_loop(self):
        self.model.train()
        for batch, data_batch in tqdm(enumerate(self.dataloader), total=len(self.dataloader)):
            data_batch_device = {key: val.to(self.device) for key, val in data_batch.items()}
            loss = self.loss_fn(self.model, data_batch_device)

            # Backpropagation
            loss.backward()
            self.optimizer.step()
            self.optimizer.zero_grad()

            if batch % 10 == 0:
                tqdm.write(f"loss: {loss.item():>7f}")

    def full_train(self, nepochs):
        for epoch in range(nepochs):
            print("########## Epoch " + str(epoch))
            self.train_loop()
            if self.scheduler is not None:
                self.scheduler.step()
    
    def save(self, path, **kwargs):
        torch.save({
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            **kwargs
            }, path)