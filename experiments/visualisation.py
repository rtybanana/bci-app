# add local modules
import sys, os
sys.path.insert(0, os.path.join(sys.path[0], '../modules'))

# imports
from preparation import prep_comp, prepall_pilot, comp_channel_map3

# constants
CLASSES = 2
FOLDS = 9
GOODS = []

# functions
def train():
  pass

def test():
  pass

# script start
# comp = prep_comp(comp_channel_map3, h_freq=30.)
# pilot = prepall_pilot(h_freq=30.)

# comp.plot()
# pilot.plot(block=True)
