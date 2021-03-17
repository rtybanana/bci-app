from .evaluate import add_kernel_dim, onehot, test_val_rest_split, get_fold, test_model
from tensorflow.python.keras.callbacks import ModelCheckpoint
from pathlib import Path
from sklearn.model_selection import train_test_split, StratifiedKFold
from .eegmodels import EEGNet

def train(model, train, validation, weight_file=None, epochs=300):
  checkpointer = ModelCheckpoint(filepath=weight_file, verbose=1, save_best_only=True) if weight_file is not None else None

  return model.fit(train['x'], train['y'], batch_size=64, epochs=epochs, verbose=0, 
                   validation_data=(validation['x'], validation['y']), callbacks=([checkpointer] if checkpointer is not None else []))


def jupyter_evaluation(train_data, test_data, classes, folds, transfer_folds, weight_path):
  _compX, _compY = train_data[0], train_data[1]
  _pilotX, _pilotY = test_data[0], test_data[1]

  chans, samples = train_data.shape[1], train_data.shape[2]
  Path(weight_path).mkdir(parents=True, exist_ok=True)

  # stratify and split pilot data for transfer learning
  skf = StratifiedKFold(transfer_folds, shuffle=True)

  comp_avg = {'acc': 0, 'bal': 0, 'kap': 0}
  pilot_avg = {'acc': 0, 'bal': 0, 'kap': 0}
  for i in range(0, folds):
    comp_trainX, comp_valX, comp_testX = add_kernel_dim(get_fold(_compX, folds, i, test_val_rest_split), kernels=1)
    comp_trainY, comp_valY, comp_testY = onehot(get_fold(_compY, folds, i, test_val_rest_split))

    # weight file path
    weight_file = f"{weight_path}/{i+1}.h5"

    # initialise model
    model = EEGNet(
      nb_classes=classes, Chans=chans, Samples=samples, 
      dropoutRate=0.5, kernLength=64, F1=8, D=2, F2=16, 
      dropoutType='Dropout'
    )

    # compile model
    model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])

    # train model and save weights
    train(model, {"x": comp_trainX, "y": comp_trainY}, {"x": comp_valX, "y": comp_valY}, weight_file, epochs=200)

    # load weights from file
    model.load_weights(weight_file)

    # test competition data
    comp_eval = test_model(model, comp_testX, comp_testY)
    
    # cross validate transfer learning
    print(f'fold {i+1}')
    pilot_fold_avg = {'acc': 0, 'bal': 0, 'kap': 0}
    for i, (train_index, test_index) in enumerate(skf.split(_pilotX, _pilotY)):
      # reset weights
      model.load_weights(weight_file)     

      # stratified train-val-test split for pilot data
      pilot_trainX, pilot_testX = _pilotX[train_index], _pilotX[test_index]
      pilot_trainY, pilot_testY = _pilotY[train_index], _pilotY[test_index]
      pilot_trainX, pilot_valX, pilot_trainY, pilot_valY = train_test_split(pilot_trainX, pilot_trainY, test_size=1/(transfer_folds-1), stratify=pilot_trainY)
      pilot_trainX, pilot_valX, pilot_testX = add_kernel_dim((pilot_trainX, pilot_valX, pilot_testX), kernels=1)
      pilot_trainY, pilot_valY, targets = onehot((pilot_trainY, pilot_valY, pilot_testY))

      # transter learn on pilot data
      train(model, {"x": pilot_trainX, "y": pilot_trainY}, {"x": pilot_valX, "y": pilot_valY}, epochs=200)

      # test pilot data
      pilot_eval = test_model(model, pilot_testX, pilot_testY)
      pilot_fold_avg = {k: pilot_fold_avg.get(k, 0) + pilot_eval.get(k, 0) for k in set(pilot_fold_avg) & set(pilot_eval)}

      print(f"\tpilot fold {i+1}:", sorted(pilot_eval.items()))

    pilot_fold_avg = {k: pilot_fold_avg.get(k, 0)/transfer_folds for k in set(pilot_fold_avg)}

    comp_avg = {k: comp_avg.get(k, 0) + comp_eval.get(k, 0) for k in set(comp_avg) & set(comp_eval)}
    pilot_avg = {k: pilot_avg.get(k, 0) + pilot_fold_avg.get(k, 0) for k in set(pilot_avg) & set(pilot_fold_avg)}

    # print evaluation
    print('competition:', sorted(comp_eval.items()))
    print('pilot:      ', sorted(pilot_fold_avg.items()))
    print()

  print('avg')
  print('competition:', sorted({k: comp_avg.get(k, 0)/folds for k in set(comp_avg)}.items()))
  print('pilot:      ', sorted({k: pilot_avg.get(k, 0)/folds for k in set(pilot_avg)}.items()))
  print()
