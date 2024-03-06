import torch
from torch import nn
from torch.utils.data import StackDataset
import hist
import awkward as ak
import matplotlib.pyplot as plt
import mplhep as hep
plt.style.use(hep.style.CMS)
from pathlib import Path
import pickle

from dnn.torch_dataset import makeSelectedInputSample, makeDataLoader
from dnn.ak_sample_loader import AkSampleLoader
from torch_scatter import scatter_add

def createHists():
    max_E, max_E_tot = 100, 800
    max_ratio = 1.5
    bins = 200

    return dict(
        h_pred = hist.Hist(hist.axis.Regular(bins, 0., max_E, name="pred_energy", label="Predicted trackster energy (GeV)")),
        h_reco = hist.Hist(hist.axis.Regular(bins, 0., max_E, name="reco_energy", label="Trackster raw energy (GeV)")),
        h_reco_tot = hist.Hist(hist.axis.Regular(bins, 0., max_E_tot, name="reco_energy_tot", label="Total trackster raw energy (GeV)")),
        h_pred_tot = hist.Hist(hist.axis.Regular(bins, 0., max_E_tot, name="pred_energy_tot", label="Predicted energy for full endcap (GeV)")),
        h_cp = hist.Hist(hist.axis.Regular(bins, 0., max_E_tot, name="cp_energy", label="CaloParticle (true) energy (GeV)")),
        h_reco_tot_over_cp = hist.Hist(hist.axis.Regular(bins, 0., max_ratio, name="reco_tot_over_cp", label="Total trackster raw energy / CaloParticle energy")),
        h_pred_tot_over_cp = hist.Hist(hist.axis.Regular(bins, 0., max_ratio, name="pred_tot_over_cp", label="Total trackster predicted energy / CaloParticle energy"))
    )

def inferenceOnSavedModel(model_path:str, model:nn.Module, input:AkSampleLoader):
    checkpoint = torch.load(model_path)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()

    pred_loader = makeDataLoader(makeSelectedInputSample(input), batch_size=200000)
    assert len(pred_loader) == 1, "Batched prediction is not yet supported"
    # model.to(pred_batch.device)

    hists = createHists()

    for pred_batch in pred_loader:
        model_output_tensor = model(pred_batch["features"]).detach()
        model_output = model_output_tensor.numpy()

        full_energy_pred = scatter_add(model_output_tensor, pred_batch["tracksterInEvent_idx"].detach()).numpy()
    
        hists["h_pred"].fill(model_output)
        hists["h_reco"].fill(ak.flatten(input.tracksters_splitEndcaps.raw_energy))
        hists["h_reco_tot"].fill(ak.sum(input.tracksters_splitEndcaps.raw_energy, axis=-1))
        hists["h_pred_tot"].fill(full_energy_pred)
        hists["h_cp"].fill(pred_batch["cp_energy"])
        hists["h_reco_tot_over_cp"].fill(ak.sum(input.tracksters_splitEndcaps.raw_energy, axis=-1) / pred_batch["cp_energy"])
        hists["h_pred_tot_over_cp"].fill(full_energy_pred / pred_batch["cp_energy"])
    
    return hists


def plotTracksterEnergies(hists):
    plt.figure(figsize=(9, 9))
    hep.histplot([hists["h_reco"], hists["h_pred"]], label=["Raw trackster energy", "Predicted trackster energy"])
    plt.ylabel("Tracksters")
    plt.xlabel("Trackster energy (GeV)")
    plt.xlim(0, 50)
    plt.legend(loc="upper right")

def plotFullEnergies(hists):
    plt.figure()
    hep.histplot([hists["h_reco_tot"], hists["h_pred_tot"], hists["h_cp"]], yerr=False, label=["Sum of raw trackster energy", "Sum of predicted energies", "CaloParticle (true) energy"])
    plt.ylabel("Events")
    plt.xlabel("Energy in endcap (GeV)")
    plt.legend()

def plotRatioOverCP(hists):
    plt.figure()
    hep.histplot([hists["h_reco_tot_over_cp"], hists["h_pred_tot_over_cp"]], yerr=False, label=["Sum of raw trackster energy", "Sum of predicted trackster energy"])
    # param_optimised,__name__ = fitCruijff(h_seedOverCP_energy)
    # x_plotFct = np.linspace(h_seedOverCP_energy.axes[0].centers[0], h_seedOverCP_energy.axes[0].centers[-1],500)
    # plt.plot(x_plotFct,cruijff(x_plotFct,*param_optimised), label=f"Cruijff fit\n$\sigma={(param_optimised[2]+param_optimised[3])/2:.3f}$")
    plt.ylabel("Events")
    plt.xlabel("Ratio over CaloParticle energy")
    plt.legend()


plotsToSave = [plotTracksterEnergies, plotFullEnergies, plotRatioOverCP]

def doFullValidation(model_path:str, model:nn.Module, input:AkSampleLoader):
    hists = inferenceOnSavedModel(model_path, model, input)
    with open(Path(model_path).with_suffix(".hists.pkl"), "wb") as f:
        pickle.dump(hists, f)
    
    for plotFct in plotsToSave:
        plotFct(hists)
        plt.savefig(Path(model_path).with_suffix("." + plotFct.__name__ + ".png"))