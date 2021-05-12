from pylsl import StreamInlet, resolve_byprop
from time import sleep
import numpy as np

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
ts = np.zeros((9, 500*seconds))
for i in range(seconds):
  sleep(1)
  chunk, timestamps = inlet.pull_chunk(max_samples=2000)
  print(chunk.shape)
    


