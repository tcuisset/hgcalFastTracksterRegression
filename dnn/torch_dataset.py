import awkward as ak
import torch
from torch.utils.data import Sampler, StackDataset, TensorDataset, Dataset, DataLoader, WeightedRandomSampler
from dataclasses import dataclass
from typing import Iterator

from dnn.ak_sample_loader import AkSampleLoader

dtype = torch.float64

@dataclass
class SelectedInputSample:
    """ Batch of inputs, as akward array """
    features:ak.Array # shape nevts * ntracksters * nfeatures * float
    tracksterInEvent_idx:ak.Array
    cp_energy:ak.Array

    def makeStackDataset(self) -> StackDataset:
        """ Make a Pytorch dataset from this batch, converting ak -> pytorch tensor """
        return StackDataset(features=FeaturesDataset(self), tracksterInEvent_idx=TracksterInEventIdxDataset(self), cp_energy=TensorDataset(torch.tensor(self.cp_energy, dtype=dtype)))

def makeSelectedInputSample(input:AkSampleLoader, featureVersion:str="feat-v1", start:int|None=None, end:int|None=None):
    doSlice = (lambda x:x[start:end]) if (start is not None and end is not None) else (lambda x:x)
    return SelectedInputSample(doSlice(input.makeDataAk(featureVersion)), doSlice(input.makeTracksterInEventIndex()), doSlice(input.caloparticles_splitEndcaps.regressed_energy))

TEST_SIZE = 10000
def makeTrainingSample(input:AkSampleLoader, featureVersion:str="feat-v1"):
    TRAIN_SIZE = len(input.tracksters_splitEndcaps) - TEST_SIZE
    return makeSelectedInputSample(input, featureVersion, 0, TRAIN_SIZE)

def makeValidationSample(input:AkSampleLoader, featureVersion:str="feat-v1"):
    TRAIN_SIZE = len(input.tracksters_splitEndcaps) - TEST_SIZE
    return makeSelectedInputSample(input, featureVersion, TRAIN_SIZE+1, len(input.tracksters_splitEndcaps))

def makeValidationAkSampleLoader(input:AkSampleLoader) -> AkSampleLoader:
    TRAIN_SIZE = len(input.tracksters_splitEndcaps) - TEST_SIZE
    return input.cloneAndSlice(TRAIN_SIZE+1, len(input.tracksters_splitEndcaps))

class FeaturesDataset(Dataset):
    def __init__(self, input:SelectedInputSample) -> None:
        self.input = input

    def __getitem__(self, index):
        return torch.tensor(self.input.features[index], dtype=dtype)

    def __len__(self):
        return len(self.input.features)

class TracksterInEventIdxDataset(Dataset):
    def __init__(self, input:SelectedInputSample) -> None:
        self.input = input

    def __getitem__(self, index):
        return torch.tensor(self.input.tracksterInEvent_idx[index], dtype=dtype)

    def __len__(self):
        return len(self.input.cp_energy)
    
def makeDataLoader(regDataset:SelectedInputSample, weighted=False, **kwargs):
    """ Create a dataloader from regression dataset. kwargs are passed to DataLoader constructor
    weighted : if True, enable sampling weights with caloparticle energy
    """
    def collate(batch):
        return {"features" : torch.cat([inp["features"] for inp in batch]), 
                "tracksterInEvent_idx":torch.cat([torch.full(inp["tracksterInEvent_idx"].shape, i) for i, inp in enumerate(batch)]), # normalize indices to start at 0 (you should not shuffle the dataset !) 
                "cp_energy":torch.stack([inp["cp_energy"][0] for inp in batch])} # not sure why [0] is needed
    sampler = None
    if weighted:
        sampler = WeightedRandomSampler(1./regDataset.cp_energy, len(regDataset.cp_energy), replacement=True)
    return DataLoader(regDataset.makeStackDataset(), collate_fn=collate, sampler=sampler, **kwargs)



