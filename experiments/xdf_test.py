## add local modules
import sys, os
sys.path.insert(0, os.path.join(sys.path[0], '../modules'))

## imports
import numpy as np
from preparation import read_raw_xdf
from mne.viz import plot_raw
from integration import GOODS, LO_FREQ, HI_FREQ

xdf = read_raw_xdf('C:/Users/The UEA VR & EEG Lab/Documents/CurrentStudy/sub-P001/ses-S001/eeg/sub-P001_ses-S001_task-Default_run-001_eeg.xdf')

to_drop = [chan for chan in xdf.ch_names if chan not in GOODS]
print(xdf.info)
xdf.drop_channels(to_drop)

tmin = 4.
xdf.crop(tmin=tmin, tmax=tmin+4)
xdf.filter(LO_FREQ, HI_FREQ, method='fir', fir_design='firwin', phase='zero')
xdf.resample(125)
print(xdf.get_data().shape)
xdf_data = xdf.get_data(start=250)*1000


for i in range(9):
    print(np.mean(xdf_data[i]))

# plot_raw(xdf, block=True)