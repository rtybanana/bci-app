# add local modules
import sys, os
sys.path.insert(0, os.path.join(sys.path[0], '../modules'))

# imports
from preparation import prep_comp, epoch_comp, prep_pilot, prepall_pilot, epoch_pilot, comp_channel_map3
from evaluation import EEGNet, get_fold, add_kernel_dim, onehot, test_val_rest_split, test

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


