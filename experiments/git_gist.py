"""
This script should throw a value error in read_epochs with a traceback similar to this:

    Traceback (most recent call last):
      File "c:/Users/roryp/vscodeprojects/bci2/scripts/git_gist.py", line 131, in <module>
        subject_1 = read_epochs('data/competition/epoched/ica/A01T-epo.fif', preload=False)
      File "<decorator-gen-210>", line 24, in read_epochs
      File "C:\Anaconda3\envs\eegnet\lib\site-packages\mne\epochs.py", line 2716, in read_epochs
        return EpochsFIF(fname, proj, preload, verbose)
      File "<decorator-gen-211>", line 24, in __init__
      File "C:\Anaconda3\envs\eegnet\lib\site-packages\mne\epochs.py", line 2778, in __init__
        _read_one_epoch_file(fid, tree, preload)
      File "C:\Anaconda3\envs\eegnet\lib\site-packages\mne\epochs.py", line 2555, in _read_one_epoch_file
        events, mappings = _read_events_fif(fid, tree)
      File "C:\Anaconda3\envs\eegnet\lib\site-packages\mne\event.py", line 156, in _read_events_fif
        raise ValueError('Could not find event data')
    ValueError: Could not find event data
"""


### loading and prepping
from mne.io import Raw, read_raw_gdf

GOODS = ['Fz','FC3','FC1','FC2','FC4','C5','C3','C1','Cz','C2','C4','C6','CP3','CP1','CPz','CP2','CP4','P1','Pz','P2','POz']
LO_FREQ = 1.
HI_FREQ = 32.
CHANNEL_MAP = {
  'EEG-Fz': 'Fz',
  'EEG-0':  'FC3',
  'EEG-1':  'FC1',
  'EEG-3':  'FC2',
  'EEG-4':  'FC4',
  'EEG-5':  'C5',
  'EEG-C3': 'C3',
  'EEG-6':  'C1',
  'EEG-Cz': 'Cz',
  'EEG-7':  'C2',
  'EEG-C4': 'C4',
  'EEG-8':  'C6',
  'EEG-9':  'CP3',
  'EEG-10': 'CP1',
  'EEG-11': 'CPz',
  'EEG-12': 'CP2',
  'EEG-13': 'CP4',
  'EEG-14': 'P1',
  'EEG-Pz': 'Pz',
  'EEG-15': 'P2',
  'EEG-16': 'POz',
  'EOG-left':'EOG-left',
  'EOG-central':'EOG-central',
  'EOG-right':'EOG-right'
}

def prep(raw: Raw, channel_map, good_channels=None, l_freq=0.5, h_freq=100.):
  """
  Performs thes basic EEG preprocessing pipeline, selecting channels and filtering
  etc. Will maybe do ICA
  """
  if isinstance(raw, list):
    raws = [[] for r in raw]
    for i, r in enumerate(raw):
      raws[i] = prep(r, channel_map, good_channels, l_freq, h_freq)
    
    return raws

  good_channels.extend(['EOG-left', 'EOG-central', 'EOG-right'])                              # add EOG channels to keep (they get dropped later during epoching)

  raw = raw.filter(l_freq, h_freq, method='fir', fir_design='firwin', phase='zero')           # bandpass filter channels between l_freq and h_freq
  raw = raw.rename_channels(channel_map)                                                      # rename channels based on passed map
  if good_channels is not None:                                                               # if any good channels provided
    raw.info['bads'] = [x for x in raw.ch_names if x not in good_channels]                    # filter channels based on passed good channels

  raw = raw.set_channel_types({'EOG-left': 'eog', 'EOG-central': 'eog', 'EOG-right': 'eog'})  # set EOG channels to EOG type because they are incorrectly marked as EEG
  raw = raw.reorder_channels(sorted(raw.ch_names))                                            # reorder channels into alphabetical
  raw = raw.set_eeg_reference(ch_type='auto')                                                 # set eeg reference to auto

  raw = raw.set_montage('standard_1020', on_missing='warn')                                   # set montage to 10/20, warn on missing

  return raw


raw = prep(read_raw_gdf('data/competition/raw/A01T.gdf', preload=True), CHANNEL_MAP, GOODS, l_freq=LO_FREQ, h_freq=HI_FREQ)


### ICA preprocessing
from mne.preprocessing import ICA

ica = ICA(n_components=21)
ica.fit(raw)
eog_indices, eog_scores = ica.find_bads_eog(raw)
ica.exclude = eog_indices
ica.apply(raw)


### epoching
from mne import Epochs, pick_types, events_from_annotations

def epoch_subjects(raw, n_classes, resample=250, trange=[-0.2, 0.5]):
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
    for i, r in enumerate(raw):
      epochses[i] = epoch_subjects(r, n_classes, resample, trange)
      
      return epochses

  # get events
  if   n_classes == 3: events, event_id = events_from_annotations(raw, event_id={'769': 0, '770': 1, '772': 2})
  elif n_classes == 2: events, event_id = events_from_annotations(raw, event_id={'769': 0, '770': 1})
  else: exit()

  # epoch creation and resampling
  picks = pick_types(raw.info, meg=False, eeg=True, stim=False, eog=False)
  epochs = Epochs(raw, events, event_id, proj=False, picks=picks, baseline=None, preload=True, verbose=False, event_repeated='merge', tmin=trange[0], tmax=trange[1])
  epochs = epochs.resample(resample)

  return epochs

epochs = epoch_subjects(raw, n_classes=3, resample=128, trange=[0.3, 2.0])


### saving
epochs.save('data/competition/epoched/ica/A01T-epo.fif', overwrite=True)


### loading
from mne import read_epochs
epochs = read_epochs('data/competition/epoched/ica/A01T-epo.fif', preload=False)
