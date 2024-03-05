import uproot
import awkward as ak
import numpy as np
import torch
from torch.utils.data import Sampler, StackDataset, TensorDataset, Dataset, DataLoader
from dataclasses import dataclass
from typing import Iterator

dtype = torch.float64

def zipTracksters(ar:ak.Array, name="trackster"):
    try:
        return ak.zip({key: ar[key] for key in ar.fields if (key not in ["event", "vertices_multiplicity", "NTracksters"]) and not ( key.startswith("EV") or key.startswith("eVector0") or key.startswith("sigmaPCA") or key.startswith("boundary"))}, depth_limit=2, with_name=name)
    except ValueError as e:
        for i in range(3, len(ar.fields)):
            try:
                ak.zip({key: ar[key] for key in ar.fields[:i] if (key not in ["event", "vertices_multiplicity", "NTracksters"]) and not ( key.startswith("EV") or key.startswith("eVector0") or key.startswith("sigmaPCA") or key.startswith("boundary"))}, depth_limit=2, with_name=name)
            except:
                raise ValueError("Field " + ar.fields[i] + " failed to be zipped", e)

def splitEndcaps(ar:ak.Array):
    return ak.concatenate([ar[ar.barycenter_eta < 0], ar[ar.barycenter_eta > 0]])


class InputSample:
    def __init__(self, pathToHisto:str, tsArgs=dict(filter_name=["raw_energy", "raw_energy_perCellType", "barycenter_*"]), cpArgs=dict(filter_name=["regressed_energy", "barycenter_*"])) -> None:
        self.dumper_tree = uproot.open(f"{pathToHisto}:ticlDumper")
        self.tracksters = self.dumper_tree["trackstersMerged"].arrays(**tsArgs)
        self.caloparticles = self.dumper_tree["simtrackstersCP"].arrays(**cpArgs)
        
        self._filterBadEvents()

        self.tracksters_splitEndcaps = splitEndcaps(zipTracksters(self.tracksters))
        self.tracksters_splitEndcaps = self.tracksters_splitEndcaps[ak.argsort(self.tracksters_splitEndcaps.raw_energy, ascending=False)]
        self.caloparticles_splitEndcaps = splitEndcaps(zipTracksters(self.caloparticles, name="caloparticle"))[:, 0]
        self._filterBadEndcaps()
        self.energyPerCellType = ak.to_regular(self.tracksters_splitEndcaps.raw_energy_perCellType, axis=-1)

    def _filterBadEvents(self):
        """ Remove events which don't have 2 caloparticles """
        good_events = (ak.num(self.caloparticles.regressed_energy) == 2)
        self.tracksters = self.tracksters[good_events]
        self.caloparticles = self.caloparticles[good_events]
    
    def _filterBadEndcaps(self):
        """ Remove endcap events which don't have any trackster reconstructed """
        good_events = (ak.num(self.tracksters_splitEndcaps.raw_energy) > 0)
        self.tracksters_splitEndcaps = self.tracksters_splitEndcaps[good_events]
        self.caloparticles_splitEndcaps = self.caloparticles_splitEndcaps[good_events]

    def makeDataAk(self):
        """ Makes an array of the features for training, output in shape nevts * ntracksters * nfeatures * float """
        return ak.concatenate([ak.unflatten(ar, counts=1, axis=-1) for ar in [self.tracksters_splitEndcaps.raw_energy, self.tracksters_splitEndcaps.barycenter_eta, self.tracksters_splitEndcaps.barycenter_z] + [self.energyPerCellType[:, :, cellType_i] for cellType_i in range(7)]], axis=-1)

    def makeTracksterInEventIndex(self):
        return ak.broadcast_arrays(ak.local_index(self.tracksters_splitEndcaps.raw_energy, axis=0), self.tracksters_splitEndcaps.raw_energy)[0]

@dataclass
class SelectedInputSample:
    features:ak.Array # shape nevts * ntracksters * nfeatures * float
    tracksterInEvent_idx:ak.Array
    cp_energy:ak.Array

    def makeStackDataset(self) -> StackDataset:
        return StackDataset(features=FeaturesDataset(self), tracksterInEvent_idx=TracksterInEventIdxDataset(self), cp_energy=TensorDataset(torch.tensor(self.cp_energy, dtype=dtype)))

def makeSelectedInputSample(input:InputSample, start:int=0, end:int=-1):
    return SelectedInputSample(input.makeDataAk()[start:end], input.makeTracksterInEventIndex()[start:end], input.caloparticles_splitEndcaps.regressed_energy[start:end])

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
    
def makeDataLoader(regDataset:SelectedInputSample, **kwargs):
    """ Create a dataloader from regression dataset. kwargs are passed to DataLoader constructor"""
    def collate(batch):
        return {"features" : torch.cat([inp["features"] for inp in batch]), 
                "tracksterInEvent_idx":torch.cat([torch.full(inp["tracksterInEvent_idx"].shape, i) for i, inp in enumerate(batch)]), # normalize indices to start at 0 (you should not shuffle the dataset !) 
                "cp_energy":torch.stack([inp["cp_energy"][0] for inp in batch])} # not sure why [0] is needed
    return DataLoader(regDataset.makeStackDataset(), collate_fn=collate, **kwargs)

