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


### script start
# _compX, _compY = epoch_comp(prep_comp(load_comp(True), comp_channel_map3, GOODS, l_freq=LO_FREQ, h_freq=HI_FREQ), CLASSES, resample=RESAMPLE, trange=T_RANGE)
_pilotX, _pilotY = epoch_pilot(load_pilot('data/rivet/raw/pilot2/BCI_imaginedmoves_3class_7-4-21.vhdr'), CLASSES, GOODS, resample=RESAMPLE, trange=T_RANGE, l_freq=LO_FREQ, h_freq=HI_FREQ)

# _compX, _compY = stratify(_compX, _compY, FOLDS)

# print(_compX.shape)
# print(_pilotX.shape)
# if (_compX.shape[2] > _pilotX.shape[2]):
#   _compX = _compX[:,:,:_pilotX.shape[2]]
# elif (_compX.shape[2] < _pilotX.shape[2]):
#   _pilotX = _pilotX[:,:,:_compX.shape[2]]

# comp_trainX, comp_valX = add_kernel_dim(get_fold(_compX, FOLDS, 0, test_rest_split), kernels=KERNELS)
# comp_trainY, comp_valY = onehot(get_fold(_compY, FOLDS, 0, test_rest_split))
# comp_valY, _ = onehot((comp_valY, []))

chans, samples = _pilotX.shape[1], _pilotX.shape[2]
Path(WEIGHT_PATH).mkdir(parents=True, exist_ok=True)

# weight file path
weight_file = f"{CLASSES}class_model.h5"

# initialise model
model = EEGNet(
  nb_classes=CLASSES, Chans=chans, Samples=samples, dropoutRate=0.5, 
  kernLength=64, F1=8, D=2, F2=16, dropoutType='Dropout'
)

# compile model
model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])

# # train model and save weights
# train(model, {"x": comp_trainX, "y": comp_trainY}, {"x": comp_valX, "y": comp_valY}, weight_file, epochs=EPOCHS)

model.load_weights(weight_file)

pilot_avg = {'acc': 0, 'bal': 0, 'kap': 0}
for i in range(REPEATS):
  # stratify and split pilot data for transfer learning
  skf = StratifiedKFold(FOLDS, shuffle=True)

  # cross validate transfer learning
  pilot_fold_avg = {'acc': 0, 'bal': 0, 'kap': 0}
  for i, (train_index, test_index) in enumerate(skf.split(_pilotX, _pilotY)):
    # reset weights
    model.load_weights(weight_file)     

    # stratified train-val-test split for pilot data
    pilot_trainX, pilot_testX = _pilotX[train_index], _pilotX[test_index]
    pilot_trainY, pilot_testY = _pilotY[train_index], _pilotY[test_index]
    pilot_trainX, pilot_valX, pilot_trainY, pilot_valY = train_test_split(pilot_trainX, pilot_trainY, test_size=1/(FOLDS-1), stratify=pilot_trainY)
    pilot_trainX, pilot_valX, pilot_testX = add_kernel_dim((pilot_trainX, pilot_valX, pilot_testX), kernels=KERNELS)
    pilot_trainY, pilot_valY, targets = onehot((pilot_trainY, pilot_valY, pilot_testY))

    # transter learn on pilot data
    train(model, {"x": pilot_trainX, "y": pilot_trainY}, {"x": pilot_valX, "y": pilot_valY}, epochs=TRANSFER_EPOCHS)

    # test pilot data
    pilot_eval = test_model_confidence(model, pilot_testX, pilot_testY, CONFIDENCE)
    pilot_fold_avg = {k: pilot_fold_avg.get(k, 0) + pilot_eval.get(k, 0) for k in set(pilot_fold_avg) & set(pilot_eval)}

    print(f"\tpilot fold {i+1}:", sorted(pilot_eval.items()))

  pilot_avg = {k: pilot_avg.get(k, 0) + pilot_fold_avg.get(k, 0) for k in set(pilot_avg) & set(pilot_fold_avg)}

  print('avg from folds')
  print('pilot:      ', sorted({k: pilot_fold_avg.get(k, 0)/FOLDS for k in set(pilot_fold_avg)}.items()))
  print()

print('avg from repeats')
print('pilot:      ', sorted({k: pilot_avg.get(k, 0)/(FOLDS * REPEATS) for k in set(pilot_avg)}.items()))