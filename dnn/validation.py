import torch
from torch import nn
from torch.utils.data import StackDataset
from torch_scatter import scatter_add
import hist
import awkward as ak
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
import mplhep as hep
plt.style.use(hep.style.CMS)
from pathlib import Path
import pickle

from dnn.torch_dataset import makeDataLoader, makeValidationSample, makeValidationAkSampleLoader, makeSelectedInputSample
from dnn.ak_sample_loader import AkSampleLoader
from dnn.fit import fitCruijff, cruijff
from dnn.model import getResultsFromModel_basicLoss

def createHists():
    max_E, max_E_tot = 100, 800
    max_ratio = 2.
    bins = 200

    return dict( # hist.axis.Regular(bins, 0., 5., name="pred_over_reco", label="Trackster predicted energy / Trackster raw energy")
        h_pred = hist.Hist(hist.axis.Regular(bins, 0., max_E, name="pred_energy", label="Predicted trackster energy (GeV)")),
        h_reco = hist.Hist(hist.axis.Regular(bins, 0., max_E, name="reco_energy", label="Trackster raw energy (GeV)")),
        h_pred_vs_reco = hist.Hist(
            hist.axis.Regular(200, 0., max_E, name="reco_energy", label="Trackster raw energy (GeV)"),
            hist.axis.Regular(200, 0., max_E, name="pred_energy", label="Predicted trackster energy (GeV)"),
            name="pred_vs_reco", label="Trackster predicted vs raw energy (GeV)"),

        h_reco_tot = hist.Hist(hist.axis.Regular(bins*3, 0., max_E_tot, name="reco_energy_tot", label="Total trackster raw energy (GeV)")),
        h_pred_tot = hist.Hist(hist.axis.Regular(bins*3, 0., max_E_tot, name="pred_energy_tot", label="Predicted energy for full endcap (GeV)")),

        h_cp = hist.Hist(hist.axis.Regular(bins*3, 0., max_E_tot, name="cp_energy", label="CaloParticle (true) energy (GeV)")),
        h_reco_tot_over_cp = hist.Hist(hist.axis.Regular(bins, 0., max_ratio, name="reco_tot_over_cp", label="Total trackster raw energy / CaloParticle energy")),
        h_pred_tot_over_cp = hist.Hist(hist.axis.Regular(bins, 0., max_ratio, name="pred_tot_over_cp", label="Total trackster predicted energy / CaloParticle energy"))
    )

def inferenceOnSavedModel(model_path:str, model:nn.Module, input:AkSampleLoader, feature_version="feat-v1", getResultsFromModel=getResultsFromModel_basicLoss, fixedEnergySample=False):
    """ Run inference on a given model, reloading state from path.len
    getResultsFromModel : function that from the output tensor of a model, returns the energy prediction
    fixedEnergySample : if False, use validation sample of training dataset. If True, use full input dataset provided
    """
    checkpoint = torch.load(model_path)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()

    if fixedEnergySample:
        pred_loader = makeDataLoader(makeSelectedInputSample(input, feature_version), batch_size=200000)
    else:
        pred_loader = makeDataLoader(makeValidationSample(input, feature_version), batch_size=200000)
        input = makeValidationAkSampleLoader(input) # get a matching ak input samples to get raw trackster energy
    assert len(pred_loader) == 1, "Batched prediction is not yet supported"
    model.to("cpu")

    hists = createHists()

    for pred_batch in pred_loader:
        pred_trackster_energies, pred_full_energy = getResultsFromModel(model, pred_batch)
    
        hists["h_pred"].fill(pred_trackster_energies)
        hists["h_reco"].fill(ak.flatten(input.tracksters_splitEndcaps.raw_energy))
        hists["h_pred_vs_reco"].fill(reco_energy=ak.flatten(input.tracksters_splitEndcaps.raw_energy), pred_energy=pred_trackster_energies)
        hists["h_reco_tot"].fill(ak.sum(input.tracksters_splitEndcaps.raw_energy, axis=-1))
        hists["h_pred_tot"].fill(pred_full_energy)
        hists["h_cp"].fill(pred_batch["cp_energy"])
        hists["h_reco_tot_over_cp"].fill(ak.sum(input.tracksters_splitEndcaps.raw_energy, axis=-1) / pred_batch["cp_energy"])
        hists["h_pred_tot_over_cp"].fill(pred_full_energy / pred_batch["cp_energy"])
    
    return hists


def plotTracksterEnergies(hists):
    plt.figure(figsize=(9, 9))
    hep.histplot([hists["h_reco"], hists["h_pred"]], label=["Raw trackster energy", "Predicted trackster energy"])
    plt.ylabel("Tracksters")
    plt.xlabel("Trackster energy (GeV)")
    plt.xlim(0, 50)
    plt.legend(loc="upper right")

import matplotlib.colors as colors
def plotTracksterRecoVsPred(hists):
    plt.figure()
    hep.hist2dplot(hists["h_pred_vs_reco"], norm=colors.LogNorm(vmin=1, vmax=hists["h_pred_vs_reco"].values().max()))

def plotTracksterRecoVsPred_profile(hists):
    plt.figure()
    hep.histplot(hists["h_pred_vs_reco"].profile("pred_energy"))
    plt.ylabel("Average predicted trackster energy (GeV)")

def plotFullEnergies(hists, xmax=None):
    plt.figure()
    hep.histplot([hists["h_reco_tot"], hists["h_pred_tot"], hists["h_cp"]], yerr=False, label=["Sum of raw trackster energy", "Sum of predicted energies", "CaloParticle (true) energy"])
    plt.ylabel("Events")
    plt.xlabel("Energy in endcap (GeV)")
    if xmax is None:
        xmax = hists["h_pred_tot"].axes[0].edges[np.max(np.nonzero(hists["h_pred_tot"].values()))+1]
    plt.xlim(left=0, right=xmax)
    plt.legend()



def plotRatioOverCP(hists):
    plt.figure()
    hep.histplot([hists["h_reco_tot_over_cp"], hists["h_pred_tot_over_cp"]], yerr=False, label=["Sum of raw trackster energy", "Sum of predicted trackster energy"])
    
    def plotFit(h:hist.Hist):
        fitRes = fitCruijff(h)
        params = fitRes.params
        x_plotFct = np.linspace(h.axes[0].centers[0], h.axes[0].centers[-1],500)
        plt.plot(x_plotFct,cruijff(x_plotFct,*params.makeTuple()), 
            label=f"Cruijff fit\n$\sigma={(params.sigmaL+params.sigmaR)/2:.3f}$, $\mu={params.m:.3f}$, " +r"$\frac{\sigma}{\mu}=" + f"{(params.sigmaL+params.sigmaR)/(2*params.m):.3f}$")

    try:
        plotFit(hists["h_reco_tot_over_cp"])
        plotFit(hists["h_pred_tot_over_cp"])
    except (ZeroDivisionError, RuntimeError) as e:
        print("Cruijff fit failed")
        print(e)

    plt.ylabel("Events")
    plt.xlabel("Ratio over CaloParticle energy")
    plt.legend()


plotsToSave = [plotTracksterEnergies, plotFullEnergies, plotRatioOverCP, plotTracksterRecoVsPred, plotTracksterRecoVsPred_profile]

def doFullValidation(model_path:str, model:nn.Module, input:AkSampleLoader, feature_version="feat-v1", getResultsFromModel=getResultsFromModel_basicLoss):
    """ Run validation on validation sample (flat energy distribution) """
    hists = inferenceOnSavedModel(model_path, model, input, feature_version, getResultsFromModel=getResultsFromModel)
    with open(Path(model_path).with_suffix(".hists.pkl"), "wb") as f:
        pickle.dump(hists, f)
    
    for plotFct in plotsToSave:
        plotFct(hists)
        plt.savefig(Path(model_path).with_suffix("." + plotFct.__name__ + ".png"))
        plt.close()

def doFullValidation_fixedEnergy(model_path:str, model:nn.Module, inputs_perEnergy:dict[float, AkSampleLoader], feature_version="feat-v1", getResultsFromModel=getResultsFromModel_basicLoss):
    """ Run validation on test samples with fixed energies 
    inputs_perEnergy : maps energy -> input test sample
    """
    for energy, input in inputs_perEnergy.items():
        hists = inferenceOnSavedModel(model_path, model, input, feature_version, getResultsFromModel=getResultsFromModel, fixedEnergySample=True)
        with open(Path(model_path).with_suffix(f".{energy}.hists.pkl"), "wb") as f:
            pickle.dump(hists, f)

        for plotFct in plotsToSave:
            plotFct(hists)
            plt.savefig(Path(model_path).with_suffix(f".{energy}." + plotFct.__name__ + ".png"))
            plt.close()