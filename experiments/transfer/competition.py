## add local modules
from re import T
import sys, os
sys.path.insert(0, os.path.join(sys.path[0], '../../modules'))

import numpy as np

## imports
from preparation import load_comp_array, epoch_comp, comp_channel_map3
from evaluation import EEGNet, get_fold, add_kernel_dim, onehot, test_rest_split, test_model, stratify
from pathlib import Path
from tensorflow.python.keras.callbacks import ModelCheckpoint
from sklearn.model_selection import train_test_split, StratifiedKFold

## constants
CLASSES = 3
FOLDS = 9
TRANSFER_FOLDS = 5
# GOODS = ['Fz','FC3','FC1','FC2','FC4','C5','C3','C1','Cz','C2','C4','C6','CP3','CP1','CPz','CP2','CP4','P1','Pz','P2','POz']
# GOODS = ['FC3','C3','CP3','FC4','C4','CP4']
GOODS = ['C3','CP3','Fz','Cz','C4','CP4']
T_RANGE = [0.5, 2.5]
RESAMPLE = 250
KERNELS = 1
EPOCHS = 100
TRANSFER_EPOCHS = 200
LO_FREQ = 4.
HI_FREQ = 38.
WEIGHT_PATH = f"weights/competition/subject-separated/{CLASSES}class/{FOLDS}fold/channel_map3"


## local functions
def train(model, train, validation, weight_file=None, epochs=300):
  checkpointer = ModelCheckpoint(filepath=weight_file, verbose=1, save_best_only=True) if weight_file is not None else None

  return model.fit(train['x'], train['y'], batch_size=32, epochs=epochs, verbose=0, 
                   validation_data=(validation['x'], validation['y']), callbacks=([checkpointer] if checkpointer is not None else []))


### script start
subject_epochs, subject_labels = epoch_comp(load_comp_array(True), CLASSES, comp_channel_map3, GOODS, RESAMPLE, T_RANGE, LO_FREQ, HI_FREQ)
print(subject_epochs)
print(subject_labels)

for i in range(0, len(subject_epochs)):
  rolled_epochs = np.roll(subject_epochs, i, 0)
  rolled_labels = np.roll(subject_labels, i, 0)

  test_epochs, test_labels = rolled_epochs[0], rolled_labels[0]
  trai_epochs, trai_labels = rolled_epochs[1:], rolled_labels[1:]

  print(test_labels)
  print(len(trai_labels))