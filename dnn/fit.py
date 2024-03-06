from functools import partial
import dataclasses
from dataclasses import dataclass

from scipy.optimize import curve_fit
import numpy as np
import hist

@dataclass
class CruijffParam:
    A:float
    """ Amplitude"""
    m:float
    """ Central value """
    sigmaL:float
    """ Left tail sigma """
    sigmaR:float
    """ Right tail sigma """
    alphaL:float
    """ Left tail alpha """
    alphaR:float
    """ Right tail alpha """

    @property
    def sigmaAverage(self) -> float:
        return (self.sigmaL + self.sigmaR) / 2
    
    def makeTuple(self) -> tuple[float]:
        return dataclasses.astuple(self)

@dataclass
class CruijffFitResult:
    params:CruijffParam
    covMatrix:np.ndarray

def cruijff(x, A, m, sigmaL,sigmaR, alphaL, alphaR):
    dx = (x-m)
    SL = np.full(x.shape, sigmaL)
    SR = np.full(x.shape, sigmaR)
    AL = np.full(x.shape, alphaL)
    AR = np.full(x.shape, alphaR)
    sigma = np.where(dx<0, SL,SR)
    alpha = np.where(dx<0, AL,AR)
    f = 2*sigma*sigma + alpha*dx*dx
    return A* np.exp(-dx*dx/f)

def fitCruijff(h_forFit:hist.Hist) -> CruijffFitResult:
    mean = np.average(h_forFit.axes[0].centers, weights=h_forFit.values())
    stdDev = np.average((h_forFit.axes[0].centers - mean)**2, weights=h_forFit.values())
    param_optimised,param_covariance_matrix = curve_fit(cruijff, h_forFit.axes[0].centers, h_forFit.values(), 
        p0=[np.max(h_forFit), mean, stdDev, stdDev,  0.1, 0.05], sigma=np.maximum(np.sqrt(h_forFit.values()), 1.8), absolute_sigma=True, maxfev=500000,
        #bounds=np.transpose([(0., np.inf), (-np.inf, np.inf), (0., np.inf), (0., np.inf), (-np.inf, np.inf), (-np.inf, np.inf)])
        )
    return CruijffFitResult(CruijffParam(*param_optimised), param_covariance_matrix)


eratio_axis = partial(hist.axis.Regular, 500, 0, 2, name="e_ratio")
eta_axis = hist.axis.Variable([1.65, 2.15, 2.75], name="absSeedEta", label="|eta|seed")
seedPt_axis = hist.axis.Variable([ 0.44310403, 11.58994007, 23.00519753, 34.58568954, 46.85866928,
       58.3225441 , 68.96975708, 80.80027771, 97.74741364], name="seedPt", label="Seed Et (GeV)") # edges are computed so that there are the same number of events in each bin
def make_scOrTsOverCP_energy_histogram(name, label=None):
    h = hist.Hist(eratio_axis(label=label),
                  eta_axis, seedPt_axis, name=name, label=label)
    return h


def fitMultiHistogram(h:hist.Hist) -> list[list[CruijffFitResult]]:
    """ Cruijff fit of multi-dimensional histogram of Supercluster/CaloParticle energy """
    res = []
    for eta_bin in range(len(h.axes["absSeedEta"])):
        res.append([])
        for seedPt_bin in range(len(h.axes["seedPt"])):
            h_1d = h[{"absSeedEta":eta_bin, "seedPt":seedPt_bin}]
            res[-1].append(fitCruijff(h_1d))
    return res

def etaBinToText(etaBin:int) -> str:
    low, high = eta_axis[etaBin]
    return r"$|\eta_{\text{seed}}| \in \left[" + f"{low}; {high}" + r"\right]$"

def ptBinToText(ptBin:int) -> str:
    low, high = seedPt_axis[ptBin]
    return r"$E_{\text{T, seed}} \in \left[" + f"{low:.3g}; {high:.3g}" + r"\right]$"