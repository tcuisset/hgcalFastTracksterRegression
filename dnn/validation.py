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

from dnn.torch_dataset import makeDataLoader, makeValidationSample, makeValidationAkSampleLoader, RegressionDataset
from dnn.ak_sample_loader import AkSampleLoader
from dnn.fit import fitCruijff, cruijff
from dnn.model import getResultsFromModel_basicLoss

def createHists():
    max_E, max_E_tot = 800, 800
    max_ratio = 2.
    bins = 400
    bins_energy = 3000
    bins_ratio = 200

    return dict( # hist.axis.Regular(bins, 0., 5., name="pred_over_reco", label="Trackster predicted energy / Trackster raw energy")
        h_pred = hist.Hist(hist.axis.Regular(bins_energy, 0., max_E, name="pred_energy", label="fastDNN Predicted trackster energy (GeV)")),
        h_reco = hist.Hist(hist.axis.Regular(bins_energy, 0., max_E, name="reco_energy", label="Trackster raw energy (GeV)")),
        h_cnn = hist.Hist(hist.axis.Regular(bins_energy, 0., max_E, name="cnn_energy", label="Trackster regressed energy (Kate CNN) (GeV)")),
        h_pred_vs_reco = hist.Hist(
            hist.axis.Regular(200, 0., max_E, name="reco_energy", label="Trackster raw energy (GeV)"),
            hist.axis.Regular(200, 0., max_E, name="pred_energy", label="Predicted trackster energy (GeV)"),
            name="pred_vs_reco", label="Trackster predicted vs raw energy (GeV)"),
        
        h_pred_vs_reco_vs_cnn = hist.Hist(
            hist.axis.Regular(200, 0., max_E, name="reco_energy", label="Trackster raw energy (GeV)"),
            hist.axis.Regular(200, 0., max_E, name="pred_energy", label="fastDNN trackster energy (GeV)"),
            hist.axis.Regular(bins, 0., max_E, name="cnn_energy", label="CNN Trackster regressed energy (Kate CNN) (GeV)"),
            name="pred_vs_reco_vs_cnn", label="Trackster predicted vs raw energy vs CNN energy (GeV)"),

        h_reco_tot = hist.Hist(hist.axis.Regular(bins_energy, 0., max_E_tot, name="reco_energy_tot", label="Total trackster raw energy (GeV)")),
        h_pred_tot = hist.Hist(hist.axis.Regular(bins_energy, 0., max_E_tot, name="pred_energy_tot", label="fastDNN energy for full endcap (GeV)")),
        h_cnn_tot = hist.Hist(hist.axis.Regular(bins_energy, 0., max_E_tot, name="cnn_energy_tot", label="CNN regressed energy for full endcap (GeV)")),

        h_cp = hist.Hist(hist.axis.Regular(bins_energy, 0., max_E_tot, name="cp_energy", label="CaloParticle (true) energy (GeV)")),
        h_reco_tot_over_cp = hist.Hist(hist.axis.Regular(bins_ratio, 0., max_ratio, name="reco_tot_over_cp", label="Total trackster raw energy / CaloParticle energy")),
        h_pred_tot_over_cp = hist.Hist(hist.axis.Regular(bins_ratio, 0., max_ratio, name="pred_tot_over_cp", label="Total trackster fastDNN energy / CaloParticle energy")),
        h_cnn_tot_over_cp = hist.Hist(hist.axis.Regular(bins_ratio, 0., max_ratio, name="cnn_tot_over_cp", label="Total trackster CNN energy / CaloParticle energy"))
    )

def fillHists(hists:dict[str, hist.Hist], cp_energy:torch.Tensor, pred_trackster_energies:torch.Tensor, pred_full_energy:torch.Tensor, raw_energy:ak.Array, raw_full_energy:ak.Array, cnn_energy:ak.Array, cnn_full_energy:ak.Array):
    hists["h_pred"].fill(pred_trackster_energies)
    hists["h_reco"].fill(raw_energy)
    hists["h_cnn"].fill(cnn_energy)

    hists["h_pred_vs_reco"].fill(reco_energy=raw_energy, pred_energy=pred_trackster_energies)
    hists["h_pred_vs_reco_vs_cnn"].fill(reco_energy=raw_energy, pred_energy=pred_trackster_energies, cnn_energy=cnn_energy)

    hists["h_reco_tot"].fill(raw_full_energy)
    hists["h_pred_tot"].fill(pred_full_energy)
    hists["h_cnn_tot"].fill(cnn_full_energy)

    hists["h_cp"].fill(cp_energy)
    hists["h_reco_tot_over_cp"].fill(raw_full_energy / cp_energy)
    hists["h_pred_tot_over_cp"].fill(pred_full_energy / cp_energy)
    hists["h_cnn_tot_over_cp"].fill(cnn_full_energy / cp_energy)

def inferenceOnSavedModel(model_path:str|None, model:nn.Module, input:AkSampleLoader, feature_version="feat-v1", getResultsFromModel=getResultsFromModel_basicLoss, fixedEnergySample=False, device="cpu"):
    """ Run inference on a given model, reloading state from mode_path
    If model_path is None, do not load checkpoint from file, assume this has already been done
    getResultsFromModel : function that from the output tensor of a model, returns the energy prediction
    fixedEnergySample : if False, use validation sample of training dataset. If True, use full input dataset provided
    """
    if model_path is not None:
        checkpoint = torch.load(model_path)
        model.load_state_dict(checkpoint["model_state_dict"])
    model.to(device)
    model.eval()

    if fixedEnergySample:
        reg_dataset = RegressionDataset(input, feature_version, device=device)
    else:
        reg_dataset = makeValidationSample(input, feature_version, device=device)
        input = makeValidationAkSampleLoader(input) # get a matching ak input samples to get raw trackster energy
    pred_loader = makeDataLoader(reg_dataset, batch_size=200000)

    assert len(pred_loader) == 1, "Batched prediction is not yet supported"

    hists = createHists()

    for pred_batch in pred_loader:
        pred_trackster_energies, pred_full_energy = getResultsFromModel(model, pred_batch)
        fillHists(hists, reg_dataset.cp_energy, pred_trackster_energies=pred_trackster_energies, pred_full_energy=pred_full_energy, 
            raw_energy=ak.flatten(input.tracksters_splitEndcaps.raw_energy), raw_full_energy=ak.sum(input.tracksters_splitEndcaps.raw_energy, axis=-1),
            cnn_energy=ak.flatten(input.tracksters_splitEndcaps.regressed_energy), cnn_full_energy=ak.sum(input.tracksters_splitEndcaps.regressed_energy, axis=-1))
        
    return hists


def plotTracksterEnergies(hists):
    plt.figure(figsize=(9, 9))
    hep.histplot([hists["h_reco"], hists["h_cnn"], hists["h_pred"]], label=["Raw energy", "CNN", "fastDNN"])
    plt.ylabel("Tracksters")
    plt.xlabel("Trackster energy (GeV)")
    plt.xlim(0, 50)
    plt.legend(loc="upper right")

import matplotlib.colors as colors
def plotTracksterRecoVsPred(hists):
    plt.figure()
    hep.hist2dplot(hists["h_pred_vs_reco"], norm=colors.LogNorm(vmin=1, vmax=hists["h_pred_vs_reco"].values().max()))
def plotTracksterRecoVsPredCNN(hists):
    plt.figure()
    h = hists["h_pred_vs_reco_vs_cnn"][{"pred_energy":hist.sum}]
    hep.hist2dplot(h, norm=colors.LogNorm(vmin=1, vmax=h.values().max()))

def plotTracksterRecoVsPred_profile(hists):
    plt.figure()
    hep.histplot([hists["h_pred_vs_reco"].profile("pred_energy"), hists["h_pred_vs_reco_vs_cnn"][{"pred_energy":hist.sum}].profile("cnn_energy")],
        label=["fastDNN", "CNN"])
    plt.ylabel("Average predicted trackster energy (GeV)")
    plt.legend()

def plotFullEnergies(hists, xmax=None, rebin=True):
    plt.figure()
    if xmax is None:
        try:
            xmax = hists["h_pred_tot"].axes[0].edges[np.max(np.nonzero(hists["h_pred_tot"].values()))+1]
        except ValueError:
            xmax = 600
    rebin_cnt = int(1+xmax/600*2) if rebin else 1
    hep.histplot([hists["h_reco_tot"][::hist.rebin(rebin_cnt)], hists["h_cnn_tot"][::hist.rebin(rebin_cnt)], hists["h_pred_tot"][::hist.rebin(rebin_cnt)], hists["h_cp"][::hist.rebin(rebin_cnt)]],
        yerr=False, label=["Sum of raw trackster energy", "Sum of CNN trackster energies", "Sum of fastDNN trackster energies", "CaloParticle (true) energy"], flow="none")
    plt.ylabel("Events")
    plt.xlabel("Energy in endcap (GeV)")

    plt.xlim(left=0, right=xmax)
    plt.legend()



def plotRatioOverCP(hists):
    plt.figure()
    hep.histplot([hists["h_reco_tot_over_cp"], hists["h_cnn_tot_over_cp"], hists["h_pred_tot_over_cp"]], yerr=False, label=["Sum of raw trackster energy", "Sum of CNN trackster energies", "Sum of fastDNN trackster energies"])
    
    def plotFit(h:hist.Hist):
        fitRes = fitCruijff(h)
        params = fitRes.params
        x_plotFct = np.linspace(h.axes[0].centers[0], h.axes[0].centers[-1],500)
        plt.plot(x_plotFct,cruijff(x_plotFct,*params.makeTuple()), 
            label=f"Cruijff fit\n$\sigma={(params.sigmaL+params.sigmaR)/2:.3f}$, $\mu={params.m:.3f}$, " +r"$\frac{\sigma}{\mu}=" + f"{(params.sigmaL+params.sigmaR)/(2*params.m):.3f}$")

    try:
        plotFit(hists["h_reco_tot_over_cp"])
        plotFit(hists["h_cnn_tot_over_cp"])
        plotFit(hists["h_pred_tot_over_cp"])
        
    except (ZeroDivisionError, RuntimeError) as e:
        print("Cruijff fit failed")
        print(e)

    plt.ylabel("Events")
    plt.xlabel("Ratio over CaloParticle energy")
    plt.legend()



plotsToSave = [plotTracksterEnergies, plotFullEnergies, plotRatioOverCP, plotTracksterRecoVsPred, plotTracksterRecoVsPredCNN, plotTracksterRecoVsPred_profile]

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


def prepareValidationDatasets_PU(inputs_perEnergy:dict[float, AkSampleLoader], feature_version="feat-v1", device="cpu"):
    return {energy : RegressionDataset(input, feature_version, device=device) for energy, input in inputs_perEnergy.items()}

default_assocScore_threshold = 0.8
def netPredictionsForCPAssociatedTracksters(input:AkSampleLoader, pred_trackster_energies, assoc_score_threshold=default_assocScore_threshold):
    # unflatten DNN output to recover event separation :
    unflat_net_res = ak.unflatten(pred_trackster_energies, ak.num(input.tracksters_splitEndcaps.raw_energy, axis=-1)) # output : nevts * ntracksters * float
    
    # compute the indices of tracksters associated with first CP
    associatedTrackstersIndices = input.assocs_simToReco_CP.tsCLUE3D_simToReco_CP[input.assocs_simToReco_CP.tsCLUE3D_simToReco_CP_score < assoc_score_threshold][:, 0, :]
    # sum the network prediction for those tracksters only
    return ak.sum(unflat_net_res[associatedTrackstersIndices], axis=-1)
def rawEnergySumForCPAssociatedTracksters(input:AkSampleLoader, branchName="raw_energy", assoc_score_threshold=default_assocScore_threshold):
    # compute the indices of tracksters associated with first CP
    associatedTrackstersIndices = input.assocs_simToReco_CP.tsCLUE3D_simToReco_CP[input.assocs_simToReco_CP.tsCLUE3D_simToReco_CP_score < assoc_score_threshold][:, 0, :]
    # sum the network prediction for those tracksters only
    return ak.sum(input.tracksters_splitEndcaps[branchName][associatedTrackstersIndices], axis=-1)



# def fillHists_PU(hists:dict[str, hist.Hist], dataset:RegressionDataset, pred_trackster_energies:torch.Tensor, pred_full_energy:torch.Tensor, raw_energy:ak.Array, raw_full_energy:ak.Array):
#     hists["h_pred"].fill(pred_trackster_energies)
#     hists["h_reco"].fill(raw_energy)
#     hists["h_pred_vs_reco"].fill(reco_energy=raw_energy, pred_energy=pred_trackster_energies)
#     hists["h_reco_tot"].fill(raw_full_energy)
#     hists["h_pred_tot"].fill(pred_full_energy)
#     hists["h_cp"].fill(dataset.cp_energy)
#     hists["h_reco_tot_over_cp"].fill(raw_full_energy / dataset.cp_energy)
#     hists["h_pred_tot_over_cp"].fill(pred_full_energy / dataset.cp_energy)


def doFullValidation_PU(pathToOutput:str, model:nn.Module, inputs_perEnergy:dict[float, AkSampleLoader], datasets_perEnergy:dict[float, RegressionDataset], getResultsFromModel=getResultsFromModel_basicLoss):
    for energy, dataset in datasets_perEnergy.items():
        pred_loader = makeDataLoader(dataset, batch_size=200000)

        pred_trackster_energies, pred_full_energy = getResultsFromModel(model, next(iter(pred_loader)))
        predForAssociatedTracksters = netPredictionsForCPAssociatedTracksters(inputs_perEnergy[energy], pred_trackster_energies)
        rawEnergyForAssociatedTracksters = rawEnergySumForCPAssociatedTracksters(inputs_perEnergy[energy])

        hists = createHists()
        fillHists(hists, dataset.cp_energy, pred_trackster_energies=pred_trackster_energies, pred_full_energy=predForAssociatedTracksters,
                  raw_energy=ak.flatten(inputs_perEnergy[energy].tracksters_splitEndcaps.raw_energy), raw_full_energy=rawEnergyForAssociatedTracksters,
                  cnn_energy=ak.flatten(inputs_perEnergy[energy].tracksters_splitEndcaps.regressed_energy), cnn_full_energy=rawEnergySumForCPAssociatedTracksters(inputs_perEnergy[energy], "regressed_energy"))

        with open(Path(pathToOutput) / f"{energy}_PU.hists.pkl", "wb") as f:
            pickle.dump(hists, f)

        for plotFct in plotsToSave:
            plotFct(hists)
            plt.savefig(Path(pathToOutput) / (f"{energy}_PU." + plotFct.__name__ + ".png"))
            plt.close()



from typing import Literal
from dnn.fit import fitCruijff, CruijffFitResult
def plotResolution(fitRes:dict[str, dict[float, CruijffFitResult]], legendLabel:dict[str, str]=None, 
              plotMode:Literal["sigmaOverMu", "sigma", "mu"]="sigmaOverMu",
              colors_datatype=['tab:blue', 'tab:red', 'tab:green', 'tab:purple'],
              errorbar_common_kwargs=dict(markeredgewidth=1.5, capsize=5, lw=1.5),
              errorbar_individual_kwargs=[ dict(fmt='.', markersize=10), dict(fmt='s', markersize=8, mfc='w'),dict(fmt='x', markersize=8)]):
    """ 
    Parameters : (typeOfData is scOverCP or tsOverCP)
     - fitRes is dict : typeOfData -> pionEnergy-> CruiffFitResult
     - legendLabel : dict : typeOfData -> legend label for typeOfData
     - plotMode : plot sigma or mu
    """
    if legendLabel is None:
        legendLabel = {typeOfData : typeOfData for typeOfData in fitRes}
    fig, main_ax = plt.subplots(figsize=(9, 8))
    main_ax.set_xlabel("Generated pion energy (GeV)")
    
    #for seedPt_bin in range(len(h.axes["seedPt"])):
    yvals_list = []
    for i, (typeOfData, currentFitResults) in enumerate(fitRes.items()):
        x_axis_centers = list(currentFitResults.keys())
        if plotMode == "sigmaOverMu":
            yvals = [res.params.sigmaAverage/res.params.m for res in currentFitResults.values()]
        elif plotMode == "mu":
            yvals = [res.params.m for res in currentFitResults.values()]
        
        main_ax.errorbar(x_axis_centers, yvals, xerr=None, label=legendLabel[typeOfData],
            **(dict(color=colors_datatype[i])|errorbar_common_kwargs|errorbar_individual_kwargs[i]))
        yvals_list.append(yvals)

        ## For the legend
        # main_ax.errorbar([], [], xerr = [], **(dict(color=colors_datatype[i])|errorbar_common_kwargs|errorbar_individual_kwargs[0]), label=etaBinToText(eta_bin))

    #for typeOfData, errorbar_kwargs in zip(fitRes.keys(), errorbar_individual_kwargs): # legend
    #    main_ax.errorbar([], [], xerr = [], **errorbar_kwargs, color='black', label=legendLabel[typeOfData])
    main_ax.legend()
    if plotMode == "sigmaOverMu":
        main_ax.set_ylabel(r'$\frac{\hat{\sigma}}{\hat{\mu}}(\frac{\sum E_{trackster}}{E_{gen}})$')
    elif plotMode == "mu":
        main_ax.set_ylabel(r'$\mu(\frac{\sum E_{trackster}}{E_{gen}})$')
    hep.cms.text("Preliminary", exp="TICLv5", ax=main_ax)
    # hep.cms.lumitext("PU=0", ax=main_ax)
    return fig