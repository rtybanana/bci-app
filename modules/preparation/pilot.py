from mne import Epochs, pick_types, events_from_annotations, concatenate_raws
from mne.io import Raw, read_raw_brainvision
from .prepare import filter_channels


def load_pilot(path):
  """
  Loads all of the data at the provided path
  """
  raw = read_raw_brainvision(path, preload=True)
  return raw


def loadall_pilot(preload=False):
  """
  Loads and concatenates the imagined and real movement pilot data
  """
  real = read_raw_brainvision('data/rivet/raw/VIPA_BCIpilot_realmovement.vhdr', preload=preload)
  imagined = read_raw_brainvision('data/rivet/raw/VIPA_BCIpilot_imaginedmovement.vhdr', preload=preload)
  raw = concatenate_raws([real, imagined])

  return raw


def epoch_pilot(raw: Raw, n_classes, good_channels, resample=250, trange=[-0.2, 0.5], l_freq=0.5, h_freq=100.):
  """
	Prepares the RIVET pilot data into epoched data depending on a passed number of 
  classes. The following event_id mapping is used:

    class       |   annotation        |   id
    left hand   |   'Stimulus/S  1'   |   0
    right hand  |   'Stimulus/S  3'   |   1
    tongue      |   'Stimulus/S  2'   |   2

  Data is resampled to 250hz after epoching to match the competition data
  """
  if isinstance(raw, list):
    epochses = []
    for i, r in enumerate(raw):
      epochses[i] = epoch_pilot(r, n_classes, good_channels, resample, trange, l_freq, h_freq)
    
    return epochses

  if   n_classes == 3: events, event_id = events_from_annotations(raw, event_id={'Stimulus/S  1': 0, 'Stimulus/S  3': 1, 'Stimulus/S  2': 2})
  elif n_classes == 2: events, event_id = events_from_annotations(raw, event_id={'Stimulus/S  1': 0, 'Stimulus/S  3': 1})
  else: exit()

  raw = raw.filter(l_freq, h_freq, method='fir', fir_design='firwin', phase='zero')
  raw = raw.notch_filter(50)
  if good_channels is not None:                                                         # if any good channels provided
    raw = filter_channels(raw, good_channels)

  raw = raw.reorder_channels(sorted(raw.ch_names))  
  raw = raw.set_eeg_reference(ch_type='auto')

  picks = pick_types(raw.info, meg=False, eeg=True, stim=False, eog=False)
  epochs = Epochs(raw, events, event_id, proj=False, picks=picks, baseline=None, preload=True, verbose=False, tmin=trange[0], tmax=trange[1])
  epochs = epochs.resample(resample)
  labels = epochs.events[:, -1]

  print(len(labels))
  print(labels)

  return (epochs.get_data()*1000, labels)


# def prep_pilot(path, good_channels, l_freq=0.5, h_freq=100.):
#   """
#   Performs the whole process of pilot data preparation to coerce it into the same
#   format as the competition data. The following steps are performed to this end:

#     1. The pilot data is loaded from the provided path
#     2. The data is band-pass filtered between 0.5hz and 100hz
#     3. The data is notch filtered at 50hz
#     4. The channels are filtered based on the provided "good" channels
#     5. The channels are reordered with respect to order of the provided "good" channels
#     6. The Raw object is returned (non-epoched)

#   The Raw object is NOT resampled at this stage because downsampling prior to epoching
#   introduces an error into the epoched data.
#   """
#   raw = load_pilot(path)
#   raw = raw.filter(l_freq, h_freq, method='fir', fir_design='firwin', phase='zero')
#   raw = raw.notch_filter(50)
#   if good_channels is not None:                                                         # if any good channels provided
#     raw = filter_channels(raw, good_channels)

#   raw = raw.reorder_channels(sorted(raw.ch_names))  
#   raw = raw.set_eeg_reference(ch_type='auto')

#   return raw


# def prepall_pilot(good_channels=None, l_freq=0.5, h_freq=100.):
#   """
#   Performs the whole process of pilot data preparation to coerce it into the same
#   format as the competition data. The following steps are performed to this end:

#     1. The pilot data is loaded from the provided path
#     2. The data is band-pass filtered between 0.5hz and 100hz
#     3. The data is notch filtered at 50hz
#     4. The channels are filtered based on the provided "good" channels
#     5. The channels are reordered with respect to order of the provided "good" channels
#     6. The Raw object is returned (non-epoched)

#   The Raw object is NOT resampled at this stage because downsampling prior to epoching
#   introduces an error into the epoched data.
#   """
#   raw = loadall_pilot()
#   raw = raw.filter(l_freq, h_freq, method='fir', fir_design='firwin', phase='zero')
#   raw = raw.notch_filter(50)
#   if good_channels is not None:
#     raw = filter_channels(raw, good_channels)
#   raw = raw.reorder_channels(sorted(raw.ch_names))  
#   raw = raw.set_eeg_reference(ch_type='auto')

#   return raw

