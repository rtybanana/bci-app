from mne import Epochs, pick_types, events_from_annotations, concatenate_raws, read_epochs
from mne.io import read_raw_gdf, Raw
from mne.channels import make_standard_montage
from .prepare import filter_channels, apply_ica
import numpy as np


def load_comp(preload=False):
  """
	Loads all of the BCI IV competition data in from the data folder in root and 
	concatenates it into a single MNE Raw data structure
  """
  raw = read_raw_gdf('data/competition/raw/A01T.gdf', preload=preload)
  for i in range(2, 10):
    print(f"\n\loading data for participant {i}")
    
    raw = concatenate_raws([raw, read_raw_gdf(f'data/competition/raw/A0{i}T.gdf', preload=preload)])

  return raw

def load_comp_array(preload=False):
  """
	Loads all of the BCI IV competition data in from the data folder in root and 
	adds each individual Raw data structure to an array so they remain separate
  """
  raws = [[] for i in range(0, 9)]
  for i in range(0, 9):
    print(f"\nloading data for participant {i+1}")
    raws[i] = read_raw_gdf(f'data/competition/raw/A0{i+1}T.gdf', preload=preload)

  return raws

def epoch_comp(raw, n_classes, resample=250, trange=[-0.2, 0.5]):
  """
	Prepares the BCIIV competition data into epoched data depending on a passed number 
  of classes. The following event_id mapping is used:

    class       |   annotation   |   id
    left hand   |   '769'        |   0
    right hand  |   '770'        |   1
    tongue      |   '772'        |   2

  If a list of raws are passed then they are each epoched using the same scheme and a
  list of Epochs objects are returned.
  """
  # if multiple raws passed
  if isinstance(raw, list):
    epochses = [[] for r in raw]
    labelses = [[] for r in raw]
    for i, r in enumerate(raw):
      epochses[i], labelses[i] = epoch_comp(r, n_classes, resample, trange)
    
    return (epochses, labelses)

  # get events
  if   n_classes == 3: events, event_id = events_from_annotations(raw, event_id={'769': 0, '770': 1, '772': 2})
  elif n_classes == 2: events, event_id = events_from_annotations(raw, event_id={'769': 0, '770': 1})
  else: exit()

  # epoch creation and resampling
  picks = pick_types(raw.info, meg=False, eeg=True, stim=False, eog=False)
  epochs = Epochs(raw, events, event_id, proj=False, picks=picks, baseline=None, preload=True, verbose=False, event_repeated='merge', tmin=trange[0], tmax=trange[1])
  epochs = epochs.resample(resample)

  labels = epochs.events[:, -1]
  return (epochs.get_data()*1000, labels)


def prep_comp(raw: Raw, channel_map, good_channels=None, l_freq=0.5, h_freq=100.):
  """
  Performs thes basic EEG preprocessing pipeline, selecting channels and filtering
  etc. Will maybe do ICA
  """
  if isinstance(raw, list):
    raws = [[] for r in raw]
    for i, r in enumerate(raw):
      raws[i] = prep_comp(r, channel_map, good_channels, l_freq, h_freq)
    
    return raws

  good_channels.extend(['EOG-left', 'EOG-central', 'EOG-right'])                              # add EOG channels to keep (they get dropped later during epoching)

  raw = raw.filter(l_freq, h_freq, method='fir', fir_design='firwin', phase='zero')           # bandpass filter channels between l_freq and h_freq
  raw = raw.rename_channels(channel_map)                                                      # rename channels based on passed map
  if good_channels is not None:                                                               # if any good channels provided
    raw = filter_channels(raw, good_channels)                                                 # filter channels based on passed good channels

  raw = raw.set_channel_types({'EOG-left': 'eog', 'EOG-central': 'eog', 'EOG-right': 'eog'})  # set EOG channels to EOG type because they are incorrectly marked as EEG
  raw = raw.reorder_channels(sorted(raw.ch_names))                                            # reorder channels into alphabetical
  raw = raw.set_eeg_reference(ch_type='auto')                                                 # set eeg reference to auto

  raw = raw.set_montage('standard_1020', on_missing='warn')                                   # set montage to 10/20, warn on missing

  return raw


def readall_comp_epochs(path):
  """
	Loads each of the 9 competition dataset participant epoch .fif files in the 
  specified directory.
  """
  epochses = [[] for i in range(0, 9)]
  for i in range(0, 9):
    print(f"\nloading data for participant {i+1}")
    print(f'{path}/A0{i+1}T-epo.fif')
    epochses[i] = read_epochs(f'{path}/A0{i+1}T-epo.fif')

  return epochses
  
