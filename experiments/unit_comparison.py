## add local modules
import sys, os
sys.path.insert(0, os.path.join(sys.path[0], '../modules'))

## imports
import numpy as np
from mne.io import read_raw_brainvision, read_raw_gdf, read_raw_fif
from preparation.constants import comp_channel_map3
from mne.viz import plot_raw
from mne.datasets import sample
from integration import GOODS, LO_FREQ, HI_FREQ

comp = read_raw_gdf('data/competition/raw/A01T.gdf', preload=True)
comp = comp.rename_channels(comp_channel_map3)
comp_drop = [chan for chan in comp.ch_names if chan not in GOODS]
comp.drop_channels(comp_drop)
comp_data = comp.get_data()

pilt = read_raw_brainvision('data/rivet/raw/pilot2/BCI_imaginedmoves_3class_7-4-21.vhdr')
pilt_drop = [chan for chan in pilt.ch_names if chan not in GOODS]
pilt.drop_channels(pilt_drop)
pilt_data = pilt.get_data()*0.1

xmpl = read_raw_fif(sample.data_path() + '/MEG/sample/sample_audvis_filt-0-40_raw.fif')
xmpl_drop = ['MEG 0111', 'MEG 0121', 'MEG 0131', 'MEG 0211', 'MEG 0221', 'MEG 0231', 'MEG 0311', 'MEG 0321', 'MEG 0331', 'MEG 1511', 'MEG 1521', 'MEG 1531']
xmpl.drop_channels(xmpl_drop)
xmpl_data = xmpl.get_data()


for i in range(9):
  print(min(comp_data[i]), max(comp_data[i]))
print()

for i in range(9):
  print(min(pilt_data[i]), max(pilt_data[i]))
print()

for i in range(9):
  print(min(xmpl_data[i]), max(xmpl_data[i]))

