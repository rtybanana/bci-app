from typing import Union
from mne import Epochs
from mne.epochs import concatenate_epochs
from mne.evoked import Evoked
from mne.io import Raw
from mne.preprocessing import ICA
from tensorflow.python.keras.backend import concatenate


def filter_channels(raw, good_channels):
  """
  Sets the mne "bad" channels based on a list of good channels
  """
  raw.info['bads'] = [x for x in raw.ch_names if x not in good_channels]
  return raw


def apply_ica(raw: Union[Raw,Epochs,Evoked], n_components=None, n_pca_components=None, random_state=None, plot=True):
  filt_raw = raw.copy()
  filt_raw.load_data().filter(l_freq=1., h_freq=None) 

  ica = ICA(n_components=n_components, n_pca_components=n_pca_components, random_state=random_state)
  ica.fit(filt_raw)

  if plot:
    raw.load_data()
    ica.plot_sources(raw)

    ica.plot_components()
  
  return ica.apply(raw)


def separateXY(epochs):
  if isinstance(epochs, list):
    epochs = concatenate_epochs(epochs)

  labels = epochs.events[:, -1]
  return (epochs.get_data()*1000, labels)
