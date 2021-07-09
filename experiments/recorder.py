import sys, os
sys.path.insert(0, os.path.join(sys.path[0], '../modules'))

from pylsl import StreamInlet, resolve_byprop
from time import sleep
import numpy as np
from integration import CHANNELS, GOODS, HI_FREQ, LO_FREQ
from mne import create_info
from mne.io import RawArray

stream_info = create_info(CHANNELS, 500, 'eeg')

def lsl_connect():
  # first resolve an EEG stream on the lab network
  print("looking for an EEG stream...")
  streams = resolve_byprop('type', 'EEG', timeout=5)

  # create a new inlet to read from the stream
  if len(streams) > 0:
    inlet = StreamInlet(streams[0])
    print('found EEG stream', inlet)
    return inlet
  else:
     return None

inlet = lsl_connect()

seconds = 20
chunk, timestamps = inlet.pull_chunk(max_samples=2000)
ts = np.zeros((0, 64))
for i in range(seconds):
  sleep(1)
  chunk, timestamps = inlet.pull_chunk(max_samples=2000)
  chunk = np.array(chunk)
  print(ts.shape)
  ts = np.concatenate([ts, chunk], axis=0)
  print(chunk.shape, ts.shape)
    

ts = ts.T
print(ts.shape)

raw = RawArray(data=ts, info=stream_info)
raw = raw.filter(LO_FREQ, HI_FREQ, method='fir', fir_design='firwin', phase='zero')
raw.resample(125)
print(raw)

raw_data = raw.get_data(picks=sorted(GOODS)) / 1000
print(raw_data.shape)

for i in range(9):
  print(min(raw_data[i]), max(raw_data[i]), np.mean(raw_data[i]))

