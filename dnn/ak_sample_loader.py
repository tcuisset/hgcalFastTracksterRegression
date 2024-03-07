import uproot
import awkward as ak
import numpy as np
import copy



def zipTracksters(ar:ak.Array, name="trackster"):
    try:
        return ak.zip({key: ar[key] for key in ar.fields if (key not in ["event", "vertices_multiplicity", "NTracksters"]) and not ( key.startswith("eVector0") or key.startswith("boundary"))}, depth_limit=2, with_name=name)
    except ValueError as e:
        for i in range(3, len(ar.fields)):
            try:
                ak.zip({key: ar[key] for key in ar.fields[:i] if (key not in ["event", "vertices_multiplicity", "NTracksters"]) and not ( key.startswith("sigmaPCA") or key.startswith("boundary"))}, depth_limit=2, with_name=name)
            except:
                raise ValueError("Field " + ar.fields[i] + " failed to be zipped", e)

def splitEndcaps(ar:ak.Array):
    return ak.concatenate([ar[ar.barycenter_eta < 0], ar[ar.barycenter_eta > 0]])



# enum :   enum CellType {
#     CE_E_120 = 0,
#     CE_E_200 = 1,
#     CE_E_300 = 2,
#     CE_H_120 = 3,
#     CE_H_200 = 4,
#     CE_H_300 = 5,
#     CE_H_SCINT = 6,
#     EnumSize = 7
#   };


class FEATURES_INDICES:
    """ Indices of features in the tensor """
    RAW_ENERGY = 0 # has to be at the same place in all feature versions, used for fraction losses

features = {
    "feat-v1" : {
        0 : "raw_energy",
        1 : "barycenter_eta",
        2 : "barycenter_z",
        3 : "energy_CE_E_120",
        4 : "energy_CE_E_200",
        5 : "energy_CE_E_300",
        6 : "energy_CE_H_120",
        7 : "energy_CE_H_200",
        8 : "energy_CE_H_300",
        9 : "energy_CE_H_SCINT"
    },
    "feat-v1-cellTypeOnly" : { # we actually use the same tensors as feat-v1 and slice them in the model itself
        0 : "energy_CE_E_120",
        1 : "energy_CE_E_200",
        2 : "energy_CE_E_300",
        3 : "energy_CE_H_120",
        4 : "energy_CE_H_200",
        5 : "energy_CE_H_300",
        6 : "energy_CE_H_SCINT"
    },
    "feat-v2" : 
        {i : feat_name for i, feat_name in enumerate(["raw_energy", "barycenter_eta", "barycenter_z", "EV1", "EV2", "EV3", "sigmaPCA1", "sigmaPCA2", "sigmaPCA3", "NClusters"])}
        | {i + 10 : feat_name for i, feat_name in enumerate(["energy_CE_E_120", "energy_CE_E_200", "energy_CE_E_300", "energy_CE_H_120", "energy_CE_H_200", "energy_CE_H_300", "energy_CE_H_SCINT"])},
}


class AkSampleLoader:
    def __init__(self, pathToHisto:str|list[str], shouldSplitEndcaps=False, cp_min_energy=1, 
                 tsArgs=dict(filter_name=["raw_energy", "raw_energy_perCellType", "barycenter_*", "EV*", "sigmaPCA*", "NClusters"]), 
                 cpArgs=dict(filter_name=["regressed_energy", "barycenter_*"])) -> None:
        """ pathToHisto : either single path, single path with wildcard, list of paths 
        shouldSplitEndcaps : if true, deals with 2 CaloParticle per event, one per each endcap. If False, only one CaloParticle per event
        cp_min_energy : remove caloparticles that have less than this energy
        """
        self.shouldSplitEndcaps = shouldSplitEndcaps
        self.cp_min_energy = cp_min_energy
        if isinstance(pathToHisto, str) and "*" not in pathToHisto:
            dumper_tree = uproot.open(f"{pathToHisto}:ticlDumper")
            self.tracksters = dumper_tree["tracksters"].arrays(**tsArgs)
            self.caloparticles = dumper_tree["simtrackstersCP"].arrays(**cpArgs)
        else:
            if isinstance(pathToHisto, str):
                pathToHisto = [pathToHisto]
            self.tracksters = uproot.concatenate([f"{fileName}:ticlDumper/tracksters" for fileName in pathToHisto], **tsArgs, num_workers=5)
            self.caloparticles = uproot.concatenate([f"{fileName}:ticlDumper/simtrackstersCP" for fileName in pathToHisto], **cpArgs, num_workers=5)
        
        self._filterBadEvents()

        maybeSplitEndcaps = splitEndcaps if shouldSplitEndcaps else lambda x:x
        self.tracksters_splitEndcaps = maybeSplitEndcaps(zipTracksters(self.tracksters))
        self.tracksters_splitEndcaps = self.tracksters_splitEndcaps[ak.argsort(self.tracksters_splitEndcaps.raw_energy, ascending=False)]
        self.caloparticles_splitEndcaps = maybeSplitEndcaps(zipTracksters(self.caloparticles, name="caloparticle"))[:, 0]
        self._filterBadEndcaps()
        self.energyPerCellType = ak.to_regular(self.tracksters_splitEndcaps.raw_energy_perCellType, axis=-1)

    def _filterBadEvents(self):
        """ Remove events which don't have 1/2 caloparticles depending on splitEndcap """
        good_events = (ak.num(self.caloparticles.regressed_energy) == (2 if self.shouldSplitEndcaps else 1))
        self.tracksters = self.tracksters[good_events]
        self.caloparticles = self.caloparticles[good_events]
    
    def _filterBadEndcaps(self):
        """ Remove endcap events which don't have any trackster reconstructed """
        good_events = (ak.num(self.tracksters_splitEndcaps.raw_energy) > 0) & (self.caloparticles_splitEndcaps.regressed_energy >= self.cp_min_energy)
        self.tracksters_splitEndcaps = self.tracksters_splitEndcaps[good_events]
        self.caloparticles_splitEndcaps = self.caloparticles_splitEndcaps[good_events]

    @classmethod
    def loadFromPickle(cls, path:str) -> "AkSampleLoader":
        import pickle
        with open(path, 'rb') as handle:
            return pickle.load(handle)
    
    def saveToPickle(self, path:str):
        import pickle
        with open(path, 'wb') as handle:
            pickle.dump(self, handle)
    
    def cloneAndSlice(self, start, end) -> "AkSampleLoader":
        """ Return a copy of self sliced according to start and end (over endcaps). Meant to match pytorch dataset splitting between train and validation """
        new = copy.copy(self)
        new.tracksters_splitEndcaps = new.tracksters_splitEndcaps[start:end]
        try:
            del new.tracksters
        except: pass
        new.caloparticles_splitEndcaps = new.caloparticles_splitEndcaps[start:end]
        try:
            del new.caloparticles
        except: pass
        new.energyPerCellType = new.energyPerCellType[start:end]
        return new
    
    def selectMainTrackster(self) -> "AkSampleLoader":
        """ Returns a copy of self removing all tracksters except the one with maximum energy per endcap """
        new = copy.copy(self)
        new.tracksters_splitEndcaps = new.tracksters_splitEndcaps[:, 0:1] # tracksters are already sorted, we use 0:1 to keep as a list
        del new.tracksters
        new.energyPerCellType = new.energyPerCellType[:, 0:1]
        return new
    
    def makeDataAk(self, features_version="feat-v1"):
        """ Makes an array of the features for training, output in shape nevts * ntracksters * nfeatures * float 
        Respects FEATURES_INDICES
        """
        energyPerCellType_list = [self.energyPerCellType[:, :, cellType_i] for cellType_i in range(7)]
        if features_version == "feat-v1" or features_version == "feat-v1-cellTypeOnly":
            return ak.concatenate([ak.unflatten(ar, counts=1, axis=-1) for ar in [self.tracksters_splitEndcaps.raw_energy, self.tracksters_splitEndcaps.barycenter_eta, self.tracksters_splitEndcaps.barycenter_z] + energyPerCellType_list], axis=-1)
        elif features_version == "feat-v2":
            return ak.concatenate(
                [ak.unflatten(self.tracksters_splitEndcaps[branch_name], counts=1, axis=-1) for branch_name in ["raw_energy", "barycenter_eta", "barycenter_z", "EV1", "EV2", "EV3", "sigmaPCA1", "sigmaPCA2", "sigmaPCA3", "NClusters"]]
                + [ak.unflatten(ar, counts=1, axis=-1) for ar in energyPerCellType_list]
            , axis=-1)
        

    def makeTracksterInEventIndex(self):
        return ak.broadcast_arrays(ak.local_index(self.tracksters_splitEndcaps.raw_energy, axis=0), self.tracksters_splitEndcaps.raw_energy)[0]
