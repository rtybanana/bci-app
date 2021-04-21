## add local modules
import sys, os
sys.path.insert(0, os.path.join(sys.path[0], '../modules'))

## imports
from mne import read_epochs
from evaluation import EEGNet, get_fold, add_kernel_dim, onehot, test_rest_split, test_model, stratify, test_model_confidence
from preparation import separateXY, load_comp, prep_comp, epoch_comp, loadall_pilot, epoch_pilot, readall_comp_epochs, comp_channel_map3, load_pilot
from pathlib import Path
from tensorflow.python.keras.callbacks import ModelCheckpoint
from sklearn.model_selection import train_test_split, StratifiedKFold

## constants
CLASSES = 3
FOLDS = 5
REPEATS = 10
GOODS = ['FC3','C3','CP3','Fz','Cz','POz','FC4','C4','CP4']
T_RANGE = [0.5, 2.5]
RESAMPLE = 125
KERNELS = 1
EPOCHS = 200
TRANSFER_EPOCHS = 300
LO_FREQ = 1.
HI_FREQ = 32.
WEIGHT_PATH = f"weights"
CONFIDENCE = 0.66


## local functions
def train(model, train, validation, weight_file=None, epochs=300):
  checkpointer = ModelCheckpoint(filepath=weight_file, verbose=1, save_best_only=True) if weight_file is not None else None

  return model.fit(train['x'], train['y'], batch_size=32, epochs=epochs, verbose=0, 
                   validation_data=(validation['x'], validation['y']), callbacks=([checkpointer] if checkpointer is not None else []))


from preparation import load_pilot
from mne import Epochs, events_from_annotations, pick_types
debug_pilot = load_pilot('data/rivet/raw/pilot2/BCI_imaginedmoves_3class_7-4-21.vhdr')
events, event_id = events_from_annotations(debug_pilot, event_id={'Stimulus/left': 0, 'Stimulus/right': 1, 'Stimulus/feet': 2})
picks = pick_types(debug_pilot.info, meg=False, eeg=True, stim=False, eog=False)
epochs = Epochs(debug_pilot, events, event_id, proj=False, picks=picks, baseline=None, preload=True, verbose=False, tmin=-10.5, tmax=2.5)
debug_labels = epochs.events[:, -1]
debug_data = epochs.get_data()
debug_data = debug_data[:,:,:-1]

### script start
# _compX, _compY = epoch_comp(prep_comp(load_comp(True), comp_channel_map3, GOODS, l_freq=LO_FREQ, h_freq=HI_FREQ), CLASSES, resample=RESAMPLE, trange=T_RANGE)
_pilotX, _pilotY = epoch_pilot(load_pilot('data/rivet/raw/pilot2/BCI_imaginedmoves_3class_7-4-21.vhdr'), CLASSES, GOODS, resample=RESAMPLE, trange=T_RANGE, l_freq=LO_FREQ, h_freq=HI_FREQ)

from mne.io import RawArray
from mne import create_info
from integration import stream_channels

raw = RawArray(data=debug_data[0], info=create_info(stream_channels, 500, 'eeg'))
raw = raw.filter(LO_FREQ, HI_FREQ, method='fir', fir_design='firwin', phase='zero')
raw = raw.notch_filter(50)
# raw = raw.drop_channels([ch for ch in raw.ch_names if ch not in GOODS])
# print(raw.ch_names)
# raw = raw.reorder_channels(sorted(raw.ch_names))
# print(raw.ch_names)
raw = raw.crop(tmin=11.)
raw = raw.resample(125)
# raw = raw.reorder_channels(sorted(raw.ch_names))

# print(raw.ch_names)
realtime = raw.get_data(picks=sorted(GOODS))*1000
raw = raw.reorder_channels(sorted(raw.ch_names))
# print(raw.ch_names)
ordered = raw.get_data(picks=GOODS)*1000

# print(realtime[0])
# print(ordered[0])

# print(_pilotX[0].shape, _pilotX[0][0])
# print(realtime.shape, realtime[0])

import numpy as np
import matplotlib.pyplot as plt
t = np.arange(0, 2, 0.008)

for i in range(9):
  for j in range(9):
    print(i, j)
    plt.plot(t, _pilotX[0][i], t, realtime[j])
    plt.show()
