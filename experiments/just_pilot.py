## add local modules
import sys, os
sys.path.insert(0, os.path.join(sys.path[0], '../modules'))

## imports
from mne import read_epochs
from evaluation import EEGNet, get_fold, add_kernel_dim, onehot, test_val_rest_split, test_model, stratify
from preparation import separateXY, load_comp, prep_comp, epoch_comp, loadall_pilot, epoch_pilot, readall_comp_epochs, comp_channel_map3
from pathlib import Path
from tensorflow.python.keras.callbacks import ModelCheckpoint
from sklearn.model_selection import train_test_split, StratifiedKFold

## constants
CLASSES = 3
FOLDS = 8
REPEATS = 10
GOODS = ['Fz','FC3','FC1','FC2','FC4','C5','C3','C1','Cz','C2','C4','C6','CP3','CP1','CPz','CP2','CP4','P1','Pz','P2','POz']
# GOODS = ['FC3','C3','CP3','FC4','C4','CP4']
# GOODS = ['FC3','C3','CP3','Fz','Cz','POz','FC4','C4','CP4']
T_RANGE = [0.5, 2.5]
RESAMPLE = 128
KERNELS = 1
EPOCHS = 100
TRANSFER_EPOCHS = 300
LO_FREQ = 1.
HI_FREQ = 32.
WEIGHT_PATH = f"weights/competition/subject-separated/{CLASSES}class/{FOLDS}fold/just_pilot"


## local functions
def train(model, train, validation, weight_file=None, epochs=300):
  checkpointer = ModelCheckpoint(filepath=weight_file, verbose=0, save_best_only=True) if weight_file is not None else None

  return model.fit(train['x'], train['y'], batch_size=16, epochs=epochs, verbose=0, 
                   validation_data=(validation['x'], validation['y']), callbacks=([checkpointer] if checkpointer is not None else []))


### script start
_pilotX, _pilotY = epoch_pilot(loadall_pilot(True), CLASSES, GOODS, resample=RESAMPLE, trange=T_RANGE, l_freq=LO_FREQ, h_freq=HI_FREQ)


chans, samples = _pilotX.shape[1], _pilotX.shape[2]
Path(WEIGHT_PATH).mkdir(parents=True, exist_ok=True)


pilot_avg = {'acc': 0, 'bal': 0, 'kap': 0}
for i in range(REPEATS):
  # stratify and split pilot data for transfer learning
  skf = StratifiedKFold(FOLDS, shuffle=True)

  pilot_fold_avg = {'acc': 0, 'bal': 0, 'kap': 0}
  for i, (train_index, test_index) in enumerate(skf.split(_pilotX, _pilotY)):
    # stratified train-val-test split for pilot data
    pilot_trainX, pilot_testX = _pilotX[train_index], _pilotX[test_index]
    pilot_trainY, pilot_testY = _pilotY[train_index], _pilotY[test_index]
    pilot_trainX, pilot_valX, pilot_trainY, pilot_valY = train_test_split(pilot_trainX, pilot_trainY, test_size=1/(FOLDS-1), stratify=pilot_trainY)
    pilot_trainX, pilot_valX, pilot_testX = add_kernel_dim((pilot_trainX, pilot_valX, pilot_testX), kernels=KERNELS)
    pilot_trainY, pilot_valY, targets = onehot((pilot_trainY, pilot_valY, pilot_testY))

    # initialise model
    model = EEGNet(
      nb_classes=CLASSES, Chans=chans, Samples=samples, dropoutRate=0.5, 
      kernLength=64, F1=8, D=2, F2=16, dropoutType='Dropout'
    )

    # compile model
    model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])

    # train model and save weights
    train(model, {"x": pilot_trainX, "y": pilot_trainY}, {"x": pilot_valX, "y": pilot_valY}, epochs=EPOCHS)

    # test pilot data
    pilot_eval = test_model(model, pilot_testX, pilot_testY)
    pilot_fold_avg = {k: pilot_fold_avg.get(k, 0) + pilot_eval.get(k, 0) for k in set(pilot_fold_avg) & set(pilot_eval)}

    print(f"\tpilot fold {i+1}:", sorted(pilot_eval.items()))

  pilot_avg = {k: pilot_avg.get(k, 0) + pilot_fold_avg.get(k, 0) for k in set(pilot_avg) & set(pilot_fold_avg)}

  print('avg from folds')
  print('pilot:      ', sorted({k: pilot_fold_avg.get(k, 0)/FOLDS for k in set(pilot_fold_avg)}.items()))


print('avg from repeats')
print('pilot:      ', sorted({k: pilot_avg.get(k, 0)/(FOLDS * REPEATS) for k in set(pilot_avg)}.items()))