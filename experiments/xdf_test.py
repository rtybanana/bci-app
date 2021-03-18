## add local modules
import sys, os
sys.path.insert(0, os.path.join(sys.path[0], '../modules'))

## imports
from preparation import read_raw_xdf
from mne.viz import plot_raw

xdf = read_raw_xdf('data/rivet/recorder/exp001/block_transfer.xdf')
print(xdf)

plot_raw(xdf, block=True)