from copy import deepcopy
import numpy as np
from scipy import stats
from tensorflow.python.keras import utils as np_utils
from tensorflow.python.keras.callbacks import ModelCheckpoint
from tensorflow.python.keras.models import Model
from sklearn.model_selection import train_test_split


def get_fold(data, n_folds, n, split_func):
  """
  Gets the nth fold by rolling the dataset by n * (len(dataset) / n_folds) 
  then returns the test-train and optional validation set by calling the 
  relevant passed function in split_func
  """
  fold_size = int(len(data)/n_folds)
  rolled = np.roll(data, n * fold_size, 0)
  return split_func(rolled, n_folds)


def test_val_rest_split(data, n_folds):
  """
  Returns a train, validation and test set by splitting the passed data into
  two parts of fold_size (calculated by len(data) / n_folds) and the rest of
  the data in the train set
  """
  fold_size = int(len(data)/n_folds)

  test  = data[0:fold_size,] 
  val   = data[fold_size:2*fold_size,]
  train = data[2*fold_size:,]

  return train, val, test


def test_rest_split(data, n_folds):
  """
  Returns a train and test set by splitting the passed data into 1 parts 
  fold_size (calculated by len(data) / n_folds) and the remaining data in 
  the train set
  """
  fold_size = int(len(data)/n_folds)
  test  = data[0:fold_size,] 
  train = data[fold_size:,]
  
  return train, test


def onehot(data_tuple):
  """
  Takes a tuple of labels as input (train, test) or (train, validation, test) 
  and returns the one hot encoded labels for train and validation in the same 
  tuple format. The test set is not one hot encoded.

  Can be chained with test_val_rest_split() or test_rest_split()
  """
  if len(data_tuple) == 3:
    return (
      np_utils.to_categorical(data_tuple[0]),
      np_utils.to_categorical(data_tuple[1]),
      data_tuple[2]
    )
  elif len(data_tuple) == 2:
    return (
      np_utils.to_categorical(data_tuple[0]),
      data_tuple[1]
    )
  elif len(data_tuple) == 1:
    return (
      data_tuple[0]
    )
  else:
    raise TypeError('Wrong number of arguments')


def add_kernel_dim(data_tuple, kernels):
  """
  Takes a tuple of labels as input (train, test) or (train, validation, test) 
  and returns a tuple of arrays reshaped to include a "kernels" parameter in 
  the 2nd axis of the train and validation sets (required by EEGNet)

  Can be chained with test_val_rest_split() or test_rest_split()
  """
  if len(data_tuple) == 3:
    return (
      data_tuple[0].reshape(data_tuple[0].shape[0], kernels, data_tuple[0].shape[1], data_tuple[0].shape[2]),
      data_tuple[1].reshape(data_tuple[1].shape[0], kernels, data_tuple[1].shape[1], data_tuple[1].shape[2]),
      data_tuple[2].reshape(data_tuple[2].shape[0], kernels, data_tuple[2].shape[1], data_tuple[2].shape[2])
    )
  elif len(data_tuple) == 2:
    return (
      data_tuple[0].reshape(data_tuple[0].shape[0], kernels, data_tuple[0].shape[1], data_tuple[0].shape[2]),
      data_tuple[1].reshape(data_tuple[1].shape[0], kernels, data_tuple[1].shape[1], data_tuple[1].shape[2])
    )
  elif len(data_tuple) == 1:
    return (
      data_tuple[0].reshape(data_tuple[0].shape[0], kernels, data_tuple[0].shape[1], data_tuple[0].shape[2])
    )

def test_model(model, test, targets):
  probs = model.predict(test)
  preds = probs.argmax(axis=-1)

  return evaluate(preds, targets)

def test_ensemble(models, test, targets):
  probs = models[0].predict(test)
  for i in range(1, len(models)):
    probs = probs + models[i].predict(test)

  preds = probs.argmax(axis=-1)
  return evaluate(preds, targets)

def test_framewise_avg(model, test, targets, n_avg_frames):
  preds = []
  for frames in test:
    to_avg = np.stack([frames[:,:,i:i - n_avg_frames] for i in range(n_avg_frames)])
    avg_probs = model.predict(to_avg)
    avg_preds = avg_probs.argmax(axis=-1)
    mode = stats.mode(avg_preds)
    print(mode[0][0])
    preds.append(mode[0][0])

  return evaluate(preds, targets)


def evaluate(preds, targets):
  import sklearn.metrics as metrics
  return {
    "acc": metrics.accuracy_score(targets, preds),
    "bal": metrics.balanced_accuracy_score(targets, preds),
    "kap": metrics.cohen_kappa_score(targets, preds)
  }

def stratify(X, y, n_folds):
  X_base, y_base = shuffle(X, y)
  idx_sort = np.argsort(y_base)
  X_base = np.array(X_base)[idx_sort]
  y_base = np.array(y_base)[idx_sort]

  fold_order = list(range(0, n_folds))
  np.random.shuffle(fold_order)

  X_folds = [[] for _ in range(0, n_folds)]
  y_folds = [[] for _ in range(0, n_folds)]
  for i, (Xi, yi) in enumerate(zip(X_base, y_base)):
    X_folds[fold_order[i % n_folds]].append(Xi)
    y_folds[fold_order[i % n_folds]].append(yi)

  X_strat = np.array([sample for fold in X_folds for sample in fold])
  y_strat = np.array([sample for fold in y_folds for sample in fold])

  return X_strat, y_strat

def shuffle(*args):
  lists = deepcopy(list(args))

  it = iter(lists)
  the_len = len(next(it))
  if not all(len(l) == the_len for l in it):
    raise ValueError('not all lists have same length!')

  rng_state = np.random.get_state()
  for l in lists:
    np.random.shuffle(l)
    np.random.set_state(rng_state)

  return tuple(lists)

def train_test_cv():
  pass

def train_test_repeat():
  pass