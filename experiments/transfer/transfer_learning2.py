## add local modules
from re import T
import sys, os
sys.path.insert(0, os.path.join(sys.path[0], '../../modules'))

## imports
from preparation import prep_comp, epoch_comp, prep_pilot, prepall_pilot, epoch_pilot, comp_channel_map3
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
T_RANGE = [-0.2, 0.5]
RESAMPLE = 250
KERNELS = 1
EPOCHS = 100
TRANSFER_EPOCHS = 200
LOW_PASS = 16.
WEIGHT_PATH = f"weights/competition/subject-separated/{CLASSES}class/{FOLDS}fold/channel_map3"


## local functions
def train(model, train, validation, weight_file=None, epochs=300):
  checkpointer = ModelCheckpoint(filepath=weight_file, verbose=1, save_best_only=True) if weight_file is not None else None

  return model.fit(train['x'], train['y'], batch_size=32, epochs=epochs, verbose=0, 
                   validation_data=(validation['x'], validation['y']), callbacks=([checkpointer] if checkpointer is not None else []))


### script start
_compX, _compY = epoch_comp(prep_comp(comp_channel_map3, GOODS, h_freq=LOW_PASS), CLASSES, resample=RESAMPLE, trange=T_RANGE)
_pilotX, _pilotY = epoch_pilot(prepall_pilot(GOODS, h_freq=LOW_PASS), CLASSES, resample=RESAMPLE, trange=T_RANGE)
# _pilotX, _pilotY = epoch_pilot(prep_pilot('data/rivet/VIPA_BCIpilot_realmovement.vhdr', GOODS, l_freq=0.5, h_freq=30.), CLASSES)

chans, samples = _pilotX.shape[1], _pilotX.shape[2]
Path(WEIGHT_PATH).mkdir(parents=True, exist_ok=True)

# stratify and split pilot data for transfer learning
skf = StratifiedKFold(TRANSFER_FOLDS, shuffle=True)

comp_avg = {'acc': 0, 'bal': 0, 'kap': 0}
pilot_avg = {'acc': 0, 'bal': 0, 'kap': 0}
for i in range(0, FOLDS):
  comp_trainX, comp_testX = get_fold(_compX, FOLDS, i, test_rest_split)
  comp_trainY, comp_testY = get_fold(_compY, FOLDS, i, test_rest_split)
  comp_trainX, comp_trainY = stratify(comp_trainX, comp_trainY, FOLDS-1)
  comp_trainX, comp_valX = get_fold(_compX, FOLDS-1, 0, test_rest_split)
  comp_trainY, comp_valY = get_fold(_compY, FOLDS-1, 0, test_rest_split)
  comp_trainX, comp_valX, comp_testX = add_kernel_dim((comp_trainX, comp_valX, comp_testX), kernels=KERNELS)
  comp_trainY, comp_valY, comp_testY = onehot((comp_trainY, comp_valY, comp_testY))


  # weight file path
  weight_file = f"{WEIGHT_PATH}/{i+1}.h5"

  # initialise model
  model = EEGNet(
    nb_classes=CLASSES, Chans=chans, Samples=samples, 
    dropoutRate=0.5, kernLength=128, F1=8, D=2, F2=16, 
    dropoutType='Dropout'
  )

  # compile model
  model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])

  # train model and save weights
  train(model, {"x": comp_trainX, "y": comp_trainY}, {"x": comp_valX, "y": comp_valY}, weight_file, epochs=EPOCHS)

  # load weights from file
  model.load_weights(weight_file)

  ### cross validate competition transfer learning
  comp_fold_avg = {'acc': 0, 'bal': 0, 'kap': 0}
  print(f'fold {i+1}')
  print('competition tl')
  for i, (train_index, test_index) in enumerate(skf.split(comp_testX, comp_testY)):
    # reset weights
    model.load_weights(weight_file)     

    # stratified train-val-test split for pilot data
    skf_comp_trainX, skf_comp_testX, skf_comp_trainY, skf_comp_testY = comp_testX[train_index], comp_testX[test_index], comp_testY[train_index], comp_testY[test_index]
    skf_comp_trainX, skf_comp_valX, skf_comp_trainY, skf_comp_valY = train_test_split(skf_comp_trainX, skf_comp_trainY, test_size=1/(TRANSFER_FOLDS-1), stratify=skf_comp_trainY)

    

    skf_comp_trainX, skf_comp_valX, skf_comp_testX = add_kernel_dim((skf_comp_trainX, skf_comp_valX, skf_comp_testX), kernels=KERNELS)
    skf_comp_trainY, skf_comp_valY, skf_comp_testY = onehot((skf_comp_trainY, skf_comp_valY, skf_comp_testY))

    # transter learn on comp data
    train(model, {"x": skf_comp_trainX, "y": skf_comp_trainY}, {"x": skf_comp_valX, "y": skf_comp_valY}, epochs=TRANSFER_EPOCHS)

    # test comp data
    comp_eval = test_model(model, skf_comp_testX, skf_comp_testY)
    comp_fold_avg = {k: comp_fold_avg.get(k, 0) + comp_eval.get(k, 0) for k in set(comp_fold_avg) & set(comp_eval)}

    print(f"\tcomp fold {i+1}:", sorted(comp_eval.items()))

  comp_fold_avg = {k: comp_fold_avg.get(k, 0)/TRANSFER_FOLDS for k in set(comp_fold_avg)}
  

  ### cross validate pilot transfer learning
  pilot_fold_avg = {'acc': 0, 'bal': 0, 'kap': 0}
  print()
  print('pilot tl')
  for i, (train_index, test_index) in enumerate(skf.split(_pilotX, _pilotY)):
    # reset weights
    model.load_weights(weight_file)     

    # stratified train-val-test split for pilot data
    pilot_trainX, pilot_testX = _pilotX[train_index], _pilotX[test_index]
    pilot_trainY, pilot_testY = _pilotY[train_index], _pilotY[test_index]
    pilot_trainX, pilot_valX, pilot_trainY, pilot_valY = train_test_split(pilot_trainX, pilot_trainY, test_size=1/(TRANSFER_FOLDS-1), stratify=pilot_trainY)
    pilot_trainX, pilot_valX, pilot_testX = add_kernel_dim((pilot_trainX, pilot_valX, pilot_testX), kernels=KERNELS)
    pilot_trainY, pilot_valY, pilot_testY = onehot((pilot_trainY, pilot_valY, pilot_testY))

    # transter learn on pilot data
    train(model, {"x": pilot_trainX, "y": pilot_trainY}, {"x": pilot_valX, "y": pilot_valY}, epochs=TRANSFER_EPOCHS)

    # test pilot data
    pilot_eval = test_model(model, pilot_testX, pilot_testY)
    pilot_fold_avg = {k: pilot_fold_avg.get(k, 0) + pilot_eval.get(k, 0) for k in set(pilot_fold_avg) & set(pilot_eval)}

    print(f"\tpilot fold {i+1}:", sorted(pilot_eval.items()))

  pilot_fold_avg = {k: pilot_fold_avg.get(k, 0)/TRANSFER_FOLDS for k in set(pilot_fold_avg)}

  comp_avg = {k: comp_avg.get(k, 0) + comp_fold_avg.get(k, 0) for k in set(comp_avg) & set(comp_fold_avg)}
  pilot_avg = {k: pilot_avg.get(k, 0) + pilot_fold_avg.get(k, 0) for k in set(pilot_avg) & set(pilot_fold_avg)}

  # print evaluation
  print('competition:', sorted(comp_fold_avg.items()))
  print('pilot:      ', sorted(pilot_fold_avg.items()))
  print()

print('avg')
print('competition:', sorted({k: comp_avg.get(k, 0)/FOLDS for k in set(comp_avg)}.items()))
print('pilot:      ', sorted({k: pilot_avg.get(k, 0)/FOLDS for k in set(pilot_avg)}.items()))
print()
