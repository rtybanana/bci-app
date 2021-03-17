## add local modules
import sys, os
sys.path.insert(0, os.path.join(sys.path[0], '../../modules'))

## imports
from preparation import prep_comp, epoch_comp
from evaluation import EEGNet, get_fold, add_kernel_dim, onehot, test_val_rest_split, test
from pathlib import Path
from tensorflow.python.keras.callbacks import ModelCheckpoint

## constants
CLASSES = 3
FOLDS = 9
TRANSFER_FOLDS = 5
# GOODS = ['Fz','FC3','FC1','FC2','FC4','C5','C3','C1','Cz','C2','C4','C6','CP3','CP1','CPz','CP2','CP4','P1','Pz','P2','POz']
GOODS = ['FC3','C3','CP3','FC4','C4','CP4']
T_RANGE = [-0.2, 0.5]
RESAMPLE = 250
KERNELS = 1
EPOCHS = 100
TRANSFER_EPOCHS = 200
HI_PASS = 1.
LO_PASS = 38.
WEIGHT_PATH = f"weights/competition/subject-separated/{CLASSES}class/{FOLDS}fold/channel_map3"


## local functions
def train(model, train, validation, weight_file):
  checkpointer = ModelCheckpoint(filepath=weight_file, verbose=1, save_best_only=True)

  return model.fit(train['x'], train['y'], batch_size=16, epochs=50, verbose=0, 
                   validation_data=(validation['x'], validation['y']), callbacks=[checkpointer])

### script start
compX, compY = epoch_comp(prep_comp({}), CLASSES)
chans, samples = compX.shape[1], compX.shape[2]

Path(WEIGHT_PATH).mkdir(parents=True, exist_ok=True)


comp_avg = {'acc': 0, 'bal': 0, 'kap': 0}
for i in range(0, FOLDS):
  trainX, valX, testX = add_kernel_dim(get_fold(compX, FOLDS, i, test_val_rest_split), kernels=KERNELS)
  trainY, valY, testY = onehot(get_fold(compY, FOLDS, i, test_val_rest_split))

  # weight file path
  weight_file = f"{WEIGHT_PATH}/{i+1}.h5"

  # initialise model
  model = EEGNet(
    nb_classes=CLASSES, Chans=chans, Samples=samples, 
    dropoutRate=0.5, kernLength=125, F1=16, D=4, F2=64, 
    dropoutType='Dropout'
  )

  # compile model
  model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])

  # train model and save weights
  # train(model, {"x": trainX, "y": trainY}, {"x": valX, "y": valY}, weight_file)

  # load weights from file
  model.load_weights(weight_file)

  # test model
  comp_eval = test(model, testX, testY)

  comp_avg = {k: comp_avg.get(k, 0) + comp_eval.get(k, 0) for k in set(comp_avg) & set(comp_eval)}

  # print evaluation
  print(f'fold {i+1}')
  print(sorted(comp_eval.items()))
  print()

print('avg')
print(sorted({k: comp_avg.get(k, 0)/FOLDS for k in set(comp_avg)}.items()))
print()
