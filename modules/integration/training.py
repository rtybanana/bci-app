from threading import Thread

from mne import concatenate_raws
from evaluation import EEGNet, stratify, test_rest_split, get_fold, add_kernel_dim, onehot
from preparation import epoch_pilot
from tensorflow.python.keras import backend as K
from tensorflow.python.keras.callbacks import ModelCheckpoint
from mne.io import read_raw_brainvision


GOODS = ['FC3','C3','CP3','Fz','Cz','POz','FC4','C4','CP4']
T_RANGE = [0.5, 2.5]
RESAMPLE = 128
EPOCHS = 300
LO_FREQ = 1.
HI_FREQ = 32
BASE_WEIGHTS = 'base_model.h5'


def train(model, train, validation, weight_file=None, epochs=300):
  checkpointer = ModelCheckpoint(filepath=weight_file, verbose=1, save_best_only=True) if weight_file is not None else None

  return model.fit(train['x'], train['y'], batch_size=16, epochs=epochs, verbose=0, 
                   validation_data=(validation['x'], validation['y']), callbacks=([checkpointer] if checkpointer is not None else []))


# class ThreadedModel(Thread):
#   def __init__(self, parent, transfer_paths=None):
#     super(ThreadedModel, self).__init__()
#     self.parent = parent
#     self.transfer_paths = transfer_paths

#   def run(self):
#     K.clear_session()

#     model = EEGNet(
#       nb_classes=3, Chans=9, Samples=256, dropoutRate=0.5, 
#       kernLength=64, F1=8, D=2, F2=16, dropoutType='Dropout'
#     )

#     # compile model loss function and optimizer for transfer learning
#     model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])

#     # load base model
#     model.load_weights(BASE_WEIGHTS)

#     # K.clear_session()

#     print(self.transfer_paths)

#     if not self.transfer_paths or len(self.transfer_paths) is 0:
#       self.parent.receive_model(model, True)
#       return

#     print(len(self.transfer_paths))
#     try:
#       transfer_raw = read_raw_brainvision(self.transfer_paths[0], preload=True)
#       if len(self.transfer_paths) > 1:
#         for i in range(1, len(self.transfer_paths)):
#           print(i)
#           transfer_raw = concatenate_raws([transfer_raw, read_raw_brainvision(self.transfer_paths[i], preload=True)])
#     except:
#       self.parent.receive_model(model, True)
#       return

#     transX, transY = epoch_pilot(transfer_raw, n_classes=3, good_channels=GOODS, resample=RESAMPLE, trange=T_RANGE, l_freq=LO_FREQ, h_freq=HI_FREQ)

#     # separate 4:1 train:validation
#     transX, transY = stratify(transX, transY, 5)

#     trans_trainX, trans_valX = add_kernel_dim(get_fold(transX, 5, 0, test_rest_split), kernels=1)
#     trans_trainY, trans_valY = onehot(get_fold(transY, 5, 0, test_rest_split))
#     trans_valY, _ = onehot((trans_valY, []))

#     # perform transfer learning on the base model and selected transfer file
#     train(model, {"x": trans_trainX, "y": trans_trainY}, {"x": trans_valX, "y": trans_valY}, epochs=EPOCHS)

#     self.parent.receive_model(model, False)


def get_model(transfer_paths=None):
  K.clear_session()

  model = EEGNet(
    nb_classes=3, Chans=9, Samples=256, dropoutRate=0.5, 
    kernLength=64, F1=8, D=2, F2=16, dropoutType='Dropout'
  )

  # compile model loss function and optimizer for transfer learning
  model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])

  # load base model
  model.load_weights(BASE_WEIGHTS)

  # K.clear_session()

  print(transfer_paths)

  if not transfer_paths or len(transfer_paths) is 0:
    return (model, True)
  print(len(transfer_paths))
  try:
    transfer_raw = read_raw_brainvision(transfer_paths[0], preload=True)
    if len(transfer_paths) > 1:
      for i in range(1, len(transfer_paths)):
        print(i)
        transfer_raw = concatenate_raws([transfer_raw, read_raw_brainvision(transfer_paths[i], preload=True)])
  except Exception as e:
    print('failed', e)
    return (model, None)

  transX, transY = epoch_pilot(transfer_raw, n_classes=3, good_channels=GOODS, resample=RESAMPLE, trange=T_RANGE, l_freq=LO_FREQ, h_freq=HI_FREQ)

  # separate 4:1 train:validation
  transX, transY = stratify(transX, transY, 5)

  trans_trainX, trans_valX = add_kernel_dim(get_fold(transX, 5, 0, test_rest_split), kernels=1)
  trans_trainY, trans_valY = onehot(get_fold(transY, 5, 0, test_rest_split))
  trans_valY, _ = onehot((trans_valY, []))

  # perform transfer learning on the base model and selected transfer file
  train(model, {"x": trans_trainX, "y": trans_trainY}, {"x": trans_valX, "y": trans_valY}, epochs=EPOCHS)

  return (model, False)
  
