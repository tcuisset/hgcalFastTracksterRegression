import awkward as ak
import torch


# Clipping tensor
TS_DIM = 3
def clipToTsDim(ar:ak.Array, fillWith:float=0.):
    return ak.fill_none(ak.pad_none(ar, TS_DIM, clip=True), fillWith)
def akToTensor(ar:ak.Array):
    return torch.tensor(clipToTsDim(ar))
def makeVarList(ar:ak.Array, vars:list[str]):
    return torch.cat([torch.tensor(clipToTsDim(ar[var])) for var in vars], dim=1)



data = torch.cat([akToTensor(tracksters_splitEndcaps.raw_energy), akToTensor(tracksters_splitEndcaps.barycenter_eta), akToTensor(tracksters_splitEndcaps.barycenter_z)] + [akToTensor(energyPerCellType[:, :, cellType_i]) for cellType_i in range(7)], dim=1)


class Network(nn.Module):
    def __init__(self):
        super().__init__()
        self.model = nn.Sequential(
            nn.Linear(data.shape[-1], 1, dtype=torch.float64),
        )

    def forward(self, x):
        return torch.squeeze(self.model(x))
net = Network()