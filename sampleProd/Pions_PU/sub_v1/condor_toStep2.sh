#!/bin/bash
set -e

export X509_USER_PROXY=/grid_mnt/vol_home/llr/cms/cuisset/.t3/proxy.cert

source /cvmfs/cms.cern.ch/cmsset_default.sh
cd /grid_mnt/vol_home/llr/cms/cuisset/hgcal/ticlRegression/CMSSW_14_0_0_pre0
cmsenv
cd - 

cat  <<EOF >>SinglePi_Pt10.py
from IOMC.RandomEngine.RandomServiceHelper import RandomNumberServiceHelper
randSvc = RandomNumberServiceHelper(process.RandomNumberGeneratorService)
randSvc.populate()

process.generator.PGunParameters.MinE = cms.double($ENERGY-0.1)
process.generator.PGunParameters.MaxE = cms.double($ENERGY+0.1)
EOF
cmsRun SinglePi_Pt10.py


cat  <<EOF >>step2.py
from IOMC.RandomEngine.RandomServiceHelper import RandomNumberServiceHelper
randSvc = RandomNumberServiceHelper(process.RandomNumberGeneratorService)
randSvc.populate()
EOF
cmsRun step2.py

mkdir -p /grid_mnt/data_cms_upgrade/cuisset/ticlRegression/PionSamples_PU/$VERSION/$ENERGY
mv step2.root /grid_mnt/data_cms_upgrade/cuisset/ticlRegression/PionSamples_PU/$VERSION/$ENERGY/step2_$INDEX.root