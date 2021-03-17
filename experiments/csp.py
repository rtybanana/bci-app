# add local modules
import sys, os
sys.path.insert(0, os.path.join(sys.path[0], '../modules'))

# imports
from preparation import prep_comp, epoch_comp, prep_pilot, prepall_pilot, epoch_pilot, comp_channel_map3
from evaluation import EEGNet, get_fold, add_kernel_dim, onehot, test_val_rest_split, test
from mne.decoding import CSP
from sklearn.svm import SVC
from sklearn.neural_network import MLPClassifier
from sklearn.model_selection import train_test_split, StratifiedKFold
from sklearn.ensemble import RandomForestClassifier

# constants
CLASSES = 2
FOLDS = 10
GOODS = ['Fz','FC3','FC1','FC2','FC4','C5','C3','C1','Cz','C2','C4','C6','CP3','CP1','CPz','CP2','CP4','P1','Pz','P2','POz']


# script start
_compX, _compY = epoch_comp(prep_comp(comp_channel_map3, GOODS, h_freq=30.), CLASSES)
_pilotX, _pilotY = epoch_pilot(prepall_pilot(GOODS, h_freq=30.), CLASSES)

csp = CSP(n_components=4)
csp.fit(_compX, _compY)
csp_compX = csp.transform(_compX)
csp_pilotX = csp.transform(_pilotX)

print(csp_compX)
print(csp_compX.shape)

# clf = SVC(kernel='linear', C=0.05, probability=True)
# clf = MLPClassifier([10, 3], batch_size=16)
clf = RandomForestClassifier()

skf = StratifiedKFold(FOLDS, shuffle=True, random_state=1)
for i, (train_index, test_index) in enumerate(skf.split(csp_compX, _compY)):
  trainX, testX = csp_compX[train_index], csp_compX[test_index]
  trainY, testY = _compY[train_index], _compY[test_index]

  clf.fit(trainX, trainY)

  preds = clf.predict(testX)
  comp_eval = test(clf, preds, testY)
  
  print(comp_eval)

