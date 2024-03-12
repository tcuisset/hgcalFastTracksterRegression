from typing import Any, Mapping
import torch
import lightning as L

import dnn.model as model
from dnn.validation import *

class FastDNNModule(L.LightningModule):
    def __init__(self, loss_fn, getResultsFromModel_fct, fixedEnergies:dict[float, RegressionDataset], fixedEnergies_PU:dict[float, RegressionDataset]) -> None:
        super().__init__()
        self.model = model.LargeDNN(17)
        self.loss_fn = loss_fn
        self.getResultsFromModel_fct = getResultsFromModel_fct
        self.fixedEnergies = fixedEnergies
        self.fixedEnergies_PU = fixedEnergies_PU

        self.hists_fixedEnergy = {energy : createHists() for energy in fixedEnergies}
        self.hists_fixedEnergy_PU = {energy : createHists() for energy in fixedEnergies_PU}
    
    def _applyModel(self, batch) -> tuple[torch.Tensor, torch.Tensor]:
        """ Returns (energy for each trackster, sum of energy for each event)"""
        return self.getResultsFromModel_fct(self.model, batch)
    
    def training_step(self, batch, batch_idx) -> torch.Tensor | Mapping[str, Any] | None:
        loss = self.loss_fn(self.model, batch)
        self.log("train_loss", loss)
        return loss
    
    def validation_step(self, batch, batch_idx):
        loss = self.loss_fn(self.model, batch)
        self.log("validation_loss", loss)
        return loss
    
    def configure_optimizers(self):
        optimizer = torch.optim.AdamW(self.model.parameters())
        return {
            "optimizer": optimizer,
            "lr_scheduler": {
                "scheduler": torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer),
                "monitor": "validation_loss",
                "frequency": 1,
                # If "monitor" references validation metrics, then "frequency" should be set to a
                # multiple of "trainer.check_val_every_n_epoch".
            },
        }
    
    def test_step(self, batch, batch_idx, dataloader_idx=0):
        pred_trackster_energies, pred_full_energy = self._applyModel(batch)
        if dataloader_idx < len(self.fixedEnergies): # fixed energy 0PU
            fillHists(self.hists_fixedEnergy[list(self.fixedEnergies.keys())[dataloader_idx]], batch["cp_energy"].detach().cpu(),
                pred_trackster_energies=pred_trackster_energies, pred_full_energy=pred_full_energy, 
                raw_energy=batch["trackster_info"][:, 0].detach().cpu().numpy(), raw_full_energy=batch["trackster_full_info"][:, 0].detach().cpu().numpy(),
                cnn_energy=batch["trackster_info"][:, 1].detach().cpu().numpy(), cnn_full_energy=batch["trackster_full_info"][:, 1].detach().cpu().numpy())
            
        elif dataloader_idx < len(self.fixedEnergies) + len(self.fixedEnergies_PU): # fixed energy 200PU
            fillHists(self.hists_fixedEnergy_PU[list(self.fixedEnergies_PU.keys())[dataloader_idx-len(self.fixedEnergies)]], batch["cp_energy"].detach().cpu(),
                pred_trackster_energies=pred_trackster_energies, pred_full_energy=pred_full_energy, 
                raw_energy=batch["trackster_info"][:, 0].detach().cpu().numpy(), raw_full_energy=batch["trackster_full_info"][:, 0].detach().cpu().numpy(),
                cnn_energy=batch["trackster_info"][:, 1].detach().cpu().numpy(), cnn_full_energy=batch["trackster_full_info"][:, 1].detach().cpu().numpy())

        else:
            assert False
    
    def on_test_end(self) -> None:
        fitRes = {}
        fitRes_CNN = {}
        fitRes_raw = {}
        for energy, hists in self.hists_fixedEnergy.items():
            with open(Path(self.logger.log_dir) / f"hists_{energy}", "wb") as f:
                pickle.dump(hists, f)
            
            def tryFit(var, h_name):
                try:
                    var[energy] = fitCruijff(hists[h_name])
                except: pass
            tryFit(fitRes, "h_pred_tot_over_cp")
            tryFit(fitRes_CNN, "h_cnn_tot_over_cp")
            tryFit(fitRes_raw, "h_reco_tot_over_cp")

            for plotFct in plotsToSave:
                plotFct(hists)
                plt.savefig(Path(self.logger.log_dir) / (f"plot_{energy}." + plotFct.__name__ + ".png"))
                self.logger.experiment.add_figure(f"Testing_0PU/{energy}_{plotFct.__name__}", plt.gcf(), self.trainer.current_epoch)
                plt.close()
        
        self.logger.experiment.add_figure("Testing_0PU/resolution", plotResolution({"raw":fitRes_raw, "cnn":fitRes_CNN, "pred":fitRes}, legendLabel=dict(raw="Raw energy", cnn="CNN", pred="fastDNN")), self.trainer.current_epoch)
        
        fitRes_PU = {}
        fitRes_CNN_PU = {}
        fitRes_raw_PU = {}
        for energy, hists in self.hists_fixedEnergy_PU.items():
            with open(Path(self.logger.log_dir) / f"hists_{energy}_PU", "wb") as f:
                pickle.dump(hists, f)
            
            def tryFit(var, h_name):
                try:
                    var[energy] = fitCruijff(hists[h_name])
                except: pass
            tryFit(fitRes_PU, "h_pred_tot_over_cp")
            tryFit(fitRes_CNN_PU, "h_cnn_tot_over_cp")
            tryFit(fitRes_raw_PU, "h_reco_tot_over_cp")

            for plotFct in plotsToSave:
                plotFct(hists)
                plt.savefig(Path(self.logger.log_dir) / (f"plot_{energy}_PU." + plotFct.__name__ + ".png"))
                self.logger.experiment.add_figure(f"Testing_200PU/{energy}_{plotFct.__name__}", plt.gcf(), self.trainer.current_epoch)
                plt.close()
        
        self.logger.experiment.add_figure("Testing_200PU/resolution", plotResolution({"raw":fitRes_raw, "cnn":fitRes_CNN, "pred":fitRes}, legendLabel=dict(raw="Raw energy", cnn="CNN", pred="fastDNN")), self.trainer.current_epoch)

        # self.logger.log_hyperparams(self.hparams, {

        # })


