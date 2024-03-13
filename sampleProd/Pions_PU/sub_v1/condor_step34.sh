#!/bin/bash
set -e

export X509_USER_PROXY=/grid_mnt/vol_home/llr/cms/cuisset/.t3/proxy.cert

source /cvmfs/cms.cern.ch/cmsset_default.sh
cd /grid_mnt/vol_home/llr/cms/cuisset/hgcal/ticlRegression/CMSSW_14_0_0_pre0
cmsenv
cd - 

cat  <<EOF >>step3.py
process.source.fileNames = cms.untracked.vstring('file:/grid_mnt/data_cms_upgrade/cuisset/ticlRegression/PionSamples_PU/$VERSION/$ENERGY/step2_$INDEX.root')
EOF

cmsRun step3.py


set +e
mv step3.root /grid_mnt/data_cms_upgrade/cuisset/ticlRegression/PionSamples_PU/$VERSION/$ENERGY/step3_$INDEX.root
mv step3_inMINIAODSIM.root /grid_mnt/data_cms_upgrade/cuisset/ticlRegression/PionSamples_PU/$VERSION/$ENERGY/step3_inMINIAODSIM_$INDEX.root
#mv step3_inDQM.root /grid_mnt/data_cms_upgrade/cuisset/ticlRegression/PionSamples_PU/$VERSION/$ENERGY/step3_inDQM_$INDEX.root
mv histo.root /grid_mnt/data_cms_upgrade/cuisset/ticlRegression/PionSamples_PU/$VERSION/$ENERGY/dumper_$INDEX.root
mv DQM_*.root /grid_mnt/data_cms_upgrade/cuisset/ticlRegression/PionSamples_PU/$VERSION/$ENERGY/DQM_$INDEX.root
