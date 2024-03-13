import awkward as ak
import numpy as np
import torch
from torch.utils.data import Sampler, StackDataset, TensorDataset, Dataset, DataLoader, WeightedRandomSampler
from dataclasses import dataclass
from typing import Iterator

from dnn.ak_sample_loader import AkSampleLoader

dtype = torch.float64



class RegressionDataset(Dataset):
    """ Batch of inputs, storing both akward array format and pytorch tensors
    One element of the dataset is one event/endcap
    """
    
    def __init__(self, input:AkSampleLoader, featureVersion:str="feat-v1", start:int|None=None, end:int|None=None, device="cpu", dtype=torch.float) -> None:
        super().__init__()
        doSlice = (lambda x:x[start:end]) if (start is not None and end is not None) else (lambda x:x)

        self.features = doSlice(input.makeDataAk(featureVersion))
        """ shape nevts * ntracksters * nfeatures * float """
        # flatten the event dimension
        self.features_tensor = torch.tensor(ak.to_numpy(ak.flatten(self.features, axis=1)), device=device, dtype=dtype)
        """ Shape (totalNbOfTsInBatch, featureCount) """

        # mapping to go from trackster to event id
        self.tracksterInEvent_idx = doSlice(input.makeTracksterInEventIndex())
        self.tracksterInEvent_idx_tensor = torch.tensor(ak.to_numpy(ak.flatten(self.tracksterInEvent_idx, axis=1)), device=device)
        """ Shape (totalNbOfTsInBatch) : maps trackster to event id """

        # mapping to go from event number to range in features_tensor and tracksterInEvent_idx (insert 0 at front)
        self.mapEventToTrackster_tensor = torch.tensor(np.insert(np.cumsum(ak.to_numpy(ak.num(doSlice(input.tracksters_splitEndcaps).raw_energy))), 0, 0), device=device)

        self.cp_energy = doSlice(input.caloparticles_splitEndcaps.regressed_energy)
        self.cp_energy_tensor = torch.tensor(ak.to_numpy(self.cp_energy), device=device, dtype=dtype)
        """ shape (nevtsInBatch) (float) """

        self.trackster_info_tensor = torch.stack([
            torch.tensor(ak.to_numpy(ak.flatten(input.tracksters_splitEndcaps.raw_energy, axis=1))),
            torch.tensor(ak.to_numpy(ak.flatten(input.tracksters_splitEndcaps.regressed_energy, axis=1)))
        ], dim=1)
        
        self.trackster_info_full_tensor = torch.stack([
            torch.tensor(ak.to_numpy(ak.sum(input.tracksters_splitEndcaps.raw_energy, axis=-1))),
            torch.tensor(ak.to_numpy(ak.sum(input.tracksters_splitEndcaps.regressed_energy, axis=-1)))
        ], dim=1)
    
    # def moveToDevice(self, device):
    #     self.features_tensor = self.features_tensor.to(device)
    #     self.tracksterInEvent_idx_tensor = self.tracksterInEvent_idx_tensor.to(device)
    #     self.mapEventToTrackster_tensor = self.mapEventToTrackster_tensor.to(device)
    #     self.cp_energy_tensor = self.cp_energy_tensor.to(device)

    def __getitem__(self, index):
        return {
            "features" : self.features_tensor[self.mapEventToTrackster_tensor[index]:self.mapEventToTrackster_tensor[index+1], :],
            "tracksterInEvent_idx" : self.tracksterInEvent_idx_tensor[self.mapEventToTrackster_tensor[index]:self.mapEventToTrackster_tensor[index+1]],
            "cp_energy" : self.cp_energy_tensor[index],

            "trackster_info" : self.trackster_info_tensor[self.mapEventToTrackster_tensor[index]:self.mapEventToTrackster_tensor[index+1], :],
            "trackster_full_info" : self.trackster_info_full_tensor[index, :],
        }

    def __len__(self):
        return len(self.features)


    # def makeStackDataset(self) -> StackDataset:
    #     """ Make a Pytorch dataset from this batch, converting ak -> pytorch tensor """
    #     return StackDataset(features=FeaturesDataset(self), tracksterInEvent_idx=TracksterInEventIdxDataset(self), cp_energy=TensorDataset(torch.tensor(self.cp_energy, dtype=dtype)))

# def RegressionDataset(input:AkSampleLoader, featureVersion:str="feat-v1", start:int|None=None, end:int|None=None):
#     doSlice = (lambda x:x[start:end]) if (start is not None and end is not None) else (lambda x:x)
#     return SelectedInputSample(doSlice(input.makeDataAk(featureVersion)), doSlice(input.makeTracksterInEventIndex()), doSlice(input.caloparticles_splitEndcaps.regressed_energy))

TEST_SIZE = 10000
def makeTrainingSample(input:AkSampleLoader, featureVersion:str="feat-v1", device="cpu"):
    TRAIN_SIZE = len(input.tracksters_splitEndcaps) - TEST_SIZE
    return RegressionDataset(input, featureVersion, 0, TRAIN_SIZE, device=device)

def makeValidationSample(input:AkSampleLoader, featureVersion:str="feat-v1", device="cpu"):
    TRAIN_SIZE = len(input.tracksters_splitEndcaps) - TEST_SIZE
    return RegressionDataset(input, featureVersion, TRAIN_SIZE+1, len(input.tracksters_splitEndcaps), device=device)

def makeValidationAkSampleLoader(input:AkSampleLoader) -> AkSampleLoader:
    TRAIN_SIZE = len(input.tracksters_splitEndcaps) - TEST_SIZE
    return input.cloneAndSlice(TRAIN_SIZE+1, len(input.tracksters_splitEndcaps))

# class FeaturesDataset(Dataset):
#     def __init__(self, input:SelectedInputSample) -> None:
#         self.input = input

#     def __getitem__(self, index):
#         return torch.tensor(self.input.features[index], dtype=dtype)

#     def __len__(self):
#         return len(self.input.features)


# class TracksterInEventIdxDataset(Dataset):
#     def __init__(self, input:SelectedInputSample) -> None:
#         self.input = input

#     def __getitem__(self, index):
#         return torch.tensor(self.input.tracksterInEvent_idx[index], dtype=dtype)

#     def __len__(self):
#         return len(self.input.cp_energy)
    
def makeDataLoader(regDataset:RegressionDataset, weighted=False, **kwargs):
    """ Create a dataloader from regression dataset. kwargs are passed to DataLoader constructor
    weighted : if True, enable sampling weights with caloparticle energy
    """
    def collate(batch):
        out = {}
        for key in batch[0].keys():
            if key == "tracksterInEvent_idx":
                out[key] = torch.cat([torch.full_like(inp["tracksterInEvent_idx"], i) for i, inp in enumerate(batch)]) # normalize indices to start at 0
            elif key == "features" or key == "trackster_info":
                out[key] = torch.cat([inp[key] for inp in batch])
            else:
                out[key] = torch.stack([inp[key] for inp in batch])
        return out
        # return {"features" : torch.cat([inp["features"] for inp in batch]), 
        #         "tracksterInEvent_idx":torch.cat([torch.full_like(inp["tracksterInEvent_idx"], i) for i, inp in enumerate(batch)]), # normalize indices to start at 0
        #         "cp_energy":torch.stack([inp["cp_energy"] for inp in batch])}
    sampler = None
    if weighted:
        sampler = WeightedRandomSampler(1./regDataset.cp_energy, len(regDataset.cp_energy), replacement=True)
    return DataLoader(regDataset, collate_fn=collate, sampler=sampler, **kwargs)



