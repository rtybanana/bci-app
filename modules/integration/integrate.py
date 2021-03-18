import threading
import time
from queue import Queue
import numpy as np

stream_channels = ['Fp1','Fz','F3','F7','F9','FC5','FC1','C3','T7','CP5','CP1','Pz','P3','P7','P9','O1','Oz','O2','P10','P8','P4','CP2','CP6','T8','C4','Cz','FC2','FC6','F10','F8','F4','Fp2','AF7','AF3','AFz','F1','F5','FT7','FC3','C1','C5','TP7','CP3','P1','P5','PO7','PO3','POz','PO4','PO8','P6','P2','CPz','CP4','TP8','C6','C2','FC4','FT8','F6','AF8','AF4','F2','Iz']

def classify(done, preds, model):
  while not done() or not trial_queue.empty():
    if not trial_queue.empty():
        trial = trial_queue.get()
        probs = model.predict(trial)
        # print(f"classified {probs.argmax()}")
        preds.append(probs.argmax())
        # socket.sendall(bytes([probs.argmax()]))
      


trial_queue = Queue()

def stream(model, test):
    print("starting stream")
    done = False
    preds = []

    x = threading.Thread(target=classify, args=(lambda: done, preds, model))
    x.start()

    for trial in test:
        try:
            trial = np.array([trial])
            # print(trial.shape, trial)
            trial_queue.put(trial)
            time.sleep(0.004)
        except KeyboardInterrupt:
            print("cancel thread")
            break

    done = True
    x.join()

    return preds
